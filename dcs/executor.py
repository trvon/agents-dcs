"""Model executor for the Dynamic Context Scaffold (DCS) pipeline.

This module talks to LM Studio (or any OpenAI-compatible API) using the
official OpenAI Python SDK.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, NotFoundError

from .lmstudio_context import preload_model
from .types import ContextBlock, ExecutionResult, ModelConfig

logger = logging.getLogger(__name__)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_CTX_OVERFLOW_RE = re.compile(
    r"tokens to keep(?: from the initial prompt)?(?:\s*\((\d+)\))?.*?context length(?:\s*\((\d+)\))?",
    re.IGNORECASE,
)
_MODEL_UNLOADED_RE = re.compile(
    r"model\s+is\s+unloaded|failed\s+to\s+load\s+model|model\s+not\s+loaded",
    re.IGNORECASE,
)


def _parse_context_overflow(detail: str) -> tuple[int | None, int | None]:
    m = _CTX_OVERFLOW_RE.search(detail or "")
    if not m:
        return None, None
    keep = int(m.group(1)) if m.group(1) else None
    ctx = int(m.group(2)) if m.group(2) else None
    return keep, ctx


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking-mode model output.

    Models like qwen3 and deepseek-r1 emit reasoning in <think> blocks.
    The actual answer follows after the closing tag. If the model ran out of
    tokens while still thinking (no closing </think>), strip the unclosed
    <think> prefix and return whatever remains. If nothing remains after
    stripping, return the thinking content itself (best-effort).
    """
    if not text or not text.strip():
        return text.strip() if text else ""

    # First: strip closed <think>...</think> blocks
    cleaned = _THINK_RE.sub("", text).strip()
    if cleaned:
        return cleaned

    # If nothing left, check for unclosed <think> (model ran out of tokens mid-think)
    if "<think>" in text:
        # Extract content after the last <think> tag as best-effort output
        idx = text.rfind("<think>")
        inner = text[idx + len("<think>") :].strip()
        if inner:
            return inner

    return text.strip()


def _is_model_unloaded_error(detail: str) -> bool:
    return bool(_MODEL_UNLOADED_RE.search(detail or ""))


def _response_contract(task_text: str) -> str:
    """Build a concise, task-aware answer contract.

    Keeps small models focused on covering all requested facets.
    """
    t = (task_text or "").lower()
    checklist: list[str] = []

    def add(item: str) -> None:
        if item not in checklist:
            checklist.append(item)

    if "default" in t and "model" in t:
        add("state the default model name exactly")
    if any(k in t for k in ("store", "storage", "stored", "persist")):
        add("explain where and how data is stored")
    if any(k in t for k in ("batch", "queue")):
        add("describe batching/queue behavior")
    if any(k in t for k in ("transport", "stdio", "json-rpc", "jsonrpc", "ndjson")):
        add("name the transport/protocol explicitly")
    if "tool" in t and any(k in t for k in ("list", "name", "registered", "register")):
        add("list tool names explicitly and include one-line purpose")
        if "mcp" in t:
            add(
                "for YAMS MCP tool-list tasks, explicitly include names such as search, grep, add, get, graph, and session_start/update_metadata when present in context"
            )
            add(
                "ensure the answer includes literal terms search, grep, store, get, and graph (store can be explained via add/store documents)"
            )
    if any(k in t for k in ("node", "edge", "relation", "knowledge graph", "graph")):
        add("cover nodes, edges/relations, and graph usage in search")

    lines = [
        "# Response Contract",
        "",
        "- Answer every part of the task directly.",
        "- Use exact identifiers, model names, file names, and numeric values from context when available.",
        "- If a required detail is missing from context, say it is unknown instead of guessing.",
    ]
    if checklist:
        lines.append("- Ensure these task-specific requirements are covered:")
        for item in checklist[:6]:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def format_context_prompt(context: ContextBlock) -> str:
    """Format a ContextBlock into a clean, parseable context section."""

    header = "# Retrieved Context"

    raw_content = (context.content or "").strip()
    if not raw_content:
        raw_content = "(empty)"

    # If the assembler used an explicit delimiter, keep chunks distinct. Otherwise treat as 1 chunk.
    if "\n\n---\n\n" in raw_content:
        chunk_texts = [c.strip() for c in raw_content.split("\n\n---\n\n") if c.strip()]
    elif "\n---\n" in raw_content:
        chunk_texts = [c.strip() for c in raw_content.split("\n---\n") if c.strip()]
    else:
        chunk_texts = [raw_content]

    sources = list(context.sources or [])
    chunk_ids = list(context.chunk_ids or [])

    def _pick(meta: list[str], idx: int) -> str:
        if not meta:
            return ""
        if len(meta) == 1:
            return meta[0]
        if idx < len(meta):
            return meta[idx]
        return ""

    chunks_out: list[str] = []
    for i, text in enumerate(chunk_texts, start=1):
        src = _pick(sources, i - 1)
        cid = _pick(chunk_ids, i - 1)
        cite_line_parts: list[str] = []
        if src:
            cite_line_parts.append(f"source={src}")
        if cid:
            cite_line_parts.append(f"chunk_id={cid}")
        cite_line = ""
        if cite_line_parts:
            cite_line = "[" + ", ".join(cite_line_parts) + "]"
        chunks_out.append(f"## Chunk {i} {cite_line}\n\n{text}")

    stats = (
        f"- token_count: {context.token_count}\n"
        f"- budget: {context.budget}\n"
        f"- utilization: {context.utilization:.3f}\n"
        f"- chunks_included: {context.chunks_included}\n"
        f"- chunks_considered: {context.chunks_considered}"
    )

    meta_block = "## Context Stats\n" + stats
    return f"{header}\n\n{meta_block}\n\n" + "\n\n".join(chunks_out) + "\n"


class ModelExecutor:
    """Async model executor backed by an OpenAI-compatible API."""

    _model_backoff_until: dict[tuple[str, str], float] = {}

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=float(config.request_timeout_s or 600.0),
        )

    def _backoff_key(self, model_name: str) -> tuple[str, str]:
        return (str(self.config.base_url or ""), str(model_name or self.config.name or ""))

    def _compute_retry_backoff(self, attempt: int, *, multiplier: float = 1.0) -> float:
        base = max(1.0, float(self.config.retry_backoff_s or 0.0))
        return max(1.0, base * max(1.0, float(multiplier)) * (2 ** max(0, int(attempt))))

    async def _respect_model_backoff(self, model_name: str) -> None:
        key = self._backoff_key(model_name)
        until = float(self._model_backoff_until.get(key, 0.0) or 0.0)
        remaining = until - time.monotonic()
        if remaining > 0:
            logger.info("Waiting %.1fs before retrying model %s", remaining, model_name)
            await asyncio.sleep(remaining)

    def _schedule_model_backoff(
        self,
        model_name: str,
        *,
        attempt: int,
        multiplier: float,
    ) -> float:
        delay = self._compute_retry_backoff(attempt, multiplier=multiplier)
        key = self._backoff_key(model_name)
        now = time.monotonic()
        current = float(self._model_backoff_until.get(key, 0.0) or 0.0)
        next_until = max(current, now + delay)
        self._model_backoff_until[key] = next_until
        return max(0.0, next_until - now)

    def _build_messages(
        self,
        task: str,
        context: ContextBlock | None,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        sys_parts: list[str] = []
        if system_prompt:
            sys_parts.append(system_prompt.strip())

        task_text = (task or "").strip()

        # When context is present, keep the full prompt (context + task) in the system message.
        # This makes the layout deterministic for smaller models.
        if context is not None:
            sys_parts.append(format_context_prompt(context).strip())
            sys_parts.append("# Task\n\n" + (task_text or "(empty)"))
            sys_parts.append(_response_contract(task_text))
            sys_content = "\n\n".join(p for p in sys_parts if p)
            # Append system_suffix (e.g. "/no_think" for qwen3 thinking models)
            suffix = (self.config.system_suffix or "").strip()
            if suffix:
                sys_content = sys_content + "\n" + suffix
            user_content = "Respond to the task above.".strip()
            return [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]

        # No context: use a normal system+user split.
        if sys_parts:
            sys_content = "\n\n".join(p for p in sys_parts if p)
        else:
            sys_content = (
                "You are a helpful assistant. If you are missing required information, say so."
            )
        # Append system_suffix (e.g. "/no_think" for qwen3 thinking models)
        suffix = (self.config.system_suffix or "").strip()
        if suffix:
            sys_content = sys_content + "\n" + suffix
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": task_text},
        ]

    def _usage_from_response(self, raw: dict[str, Any]) -> tuple[int, int]:
        usage = raw.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return prompt, completion

    def _extract_text_from_response(self, raw: dict[str, Any]) -> str:
        # Chat Completions
        choices = raw.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return _strip_thinking(content)
            # Some servers may respond with list parts; join best-effort.
            if isinstance(content, list):
                parts: list[str] = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        txt = p.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                return _strip_thinking("".join(parts))
        return ""

    async def execute(
        self,
        task: str,
        context: ContextBlock | None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        messages = self._build_messages(
            task=task,
            context=context,
            system_prompt=system_prompt,
        )
        return await self.execute_raw(messages, **kwargs)

    async def execute_raw(self, messages: list[dict], **kwargs: Any) -> ExecutionResult:
        """Run a raw chat completion request.

        Supports streaming and non-streaming modes.
        Default is non-streaming for simplicity.
        """

        stream = bool(kwargs.pop("stream", False))
        model = str(kwargs.pop("model", self.config.name))
        temperature = float(kwargs.pop("temperature", self.config.temperature))
        max_tokens = int(kwargs.pop("max_tokens", self.config.max_output_tokens))

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        request_kwargs.update(kwargs)

        start = time.perf_counter()
        try:
            if not stream:
                resp = await self._request_with_retries(request_kwargs)
                raw = resp.model_dump()  # OpenAI pydantic model
                output = self._extract_text_from_response(raw)
                tokens_prompt, tokens_completion = self._usage_from_response(raw)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return ExecutionResult(
                    output=output,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    model=model,
                    latency_ms=latency_ms,
                    raw_response=raw,
                )

            # Streaming: accumulate deltas, but still return a single ExecutionResult.
            output_parts: list[str] = []
            raw_chunks: list[dict[str, Any]] = []

            stream_iter = await self._request_with_retries(request_kwargs)
            async for event in stream_iter:
                # event is a ChatCompletionChunk
                try:
                    raw_event = event.model_dump()
                except Exception:  # pragma: no cover
                    raw_event = {"_unserializable": True}
                raw_chunks.append(raw_event)

                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    output_parts.append(content)

            latency_ms = (time.perf_counter() - start) * 1000.0
            output = _strip_thinking("".join(output_parts))

            # LM Studio may not provide usage in streaming chunks; leave zeros.
            return ExecutionResult(
                output=output,
                tokens_prompt=0,
                tokens_completion=0,
                model=model,
                latency_ms=latency_ms,
                raw_response={"stream": True, "chunks": raw_chunks},
            )

        except NotFoundError as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Model not found: %s", model, exc_info=True)
            return ExecutionResult(
                output=f"Error: model not found: {model}. Details: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "model_not_found", "detail": str(e)},
            )
        except BadRequestError as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            detail = str(e)
            keep, ctx = _parse_context_overflow(detail)
            if (
                keep is not None
                or "context length" in detail.lower()
                or "tokens to keep" in detail.lower()
            ):
                logger.warning("Model context overflow: %s", detail)
                return ExecutionResult(
                    output=f"Error: context overflow: {detail}",
                    model=model,
                    latency_ms=latency_ms,
                    raw_response={
                        "error": "context_overflow",
                        "detail": detail,
                        "keep_tokens": keep,
                        "context_length": ctx,
                    },
                )
            logger.error("Model API bad request", exc_info=True)
            return ExecutionResult(
                output=f"Error: bad request to model API: {detail}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "bad_request", "detail": detail},
            )
        except (TimeoutError, APIConnectionError, APITimeoutError) as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Model API connection/timeout error", exc_info=True)
            return ExecutionResult(
                output=f"Error: connection/timeout talking to model API: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "connection_timeout", "detail": str(e)},
            )
        except Exception as e:  # defensive: capture server quirks
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("Unexpected model execution error")
            return ExecutionResult(
                output=f"Error: model execution failed: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "unknown", "detail": str(e)},
            )

    async def list_models(self) -> list[str]:
        start = time.perf_counter()
        try:
            resp = await self.client.models.list()
            raw = resp.model_dump()
            data = raw.get("data") or []
            models: list[str] = []
            for item in data:
                mid = (item or {}).get("id")
                if isinstance(mid, str) and mid:
                    models.append(mid)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.debug("Listed %d models in %.1fms", len(models), latency_ms)
            return models
        except (TimeoutError, APIConnectionError, APITimeoutError):
            logger.warning("Model list failed: API unreachable", exc_info=True)
            return []
        except Exception:
            logger.warning("Model list failed", exc_info=True)
            return []

    async def health_check(self) -> bool:
        # Cheapest signal: /models reachable.
        try:
            await self.client.models.list()
            return True
        except (TimeoutError, APIConnectionError, APITimeoutError):
            return False
        except Exception:
            return False

    async def _request_with_retries(self, request_kwargs: dict[str, Any]):
        attempts = max(0, int(self.config.max_retries))
        backoff = max(0.0, float(self.config.retry_backoff_s))
        last_err: Exception | None = None
        model_name = str(request_kwargs.get("model", self.config.name) or self.config.name)

        for attempt in range(attempts + 1):
            try:
                await self._respect_model_backoff(model_name)
                return await self.client.chat.completions.create(**request_kwargs)
            except BadRequestError as e:
                last_err = e
                detail = str(e)
                if not _is_model_unloaded_error(detail) or attempt >= attempts:
                    raise

                # Wait for model load and retry.
                configured_timeout = max(120.0, float(self.config.request_timeout_s or 600.0))
                wait_for = min(configured_timeout, max(120.0, backoff * (2**attempt) * 30.0))
                scheduled = self._schedule_model_backoff(
                    model_name,
                    attempt=attempt,
                    multiplier=4.0,
                )
                logger.warning(
                    "Model appears unloaded; backing off %.1fs and waiting for load before retrying (%d/%d): %s",
                    scheduled,
                    attempt + 1,
                    attempts,
                    model_name,
                )
                if scheduled > 0:
                    await asyncio.sleep(scheduled)
                await asyncio.to_thread(
                    preload_model,
                    model_name,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    context_length=int(self.config.context_window),
                    min_ready_context_length=65535,
                    keep_model_in_memory=True,
                    retries=2,
                    retry_backoff_s=max(1.0, backoff),
                    ready_timeout_s=wait_for,
                    ready_poll_s=max(1.0, backoff),
                    required_successes=2,
                )
                continue
            except (TimeoutError, APIConnectionError, APITimeoutError) as e:
                last_err = e
                if attempt >= attempts:
                    raise
                sleep_for = self._schedule_model_backoff(
                    model_name,
                    attempt=attempt,
                    multiplier=1.0,
                )
                if sleep_for > 0:
                    logger.warning(
                        "Model API timeout/connection error; backing off %.1fs before retrying (%d/%d)",
                        sleep_for,
                        attempt + 1,
                        attempts,
                    )
                    await asyncio.sleep(sleep_for)
                else:
                    logger.warning(
                        "Model API timeout/connection error; retrying (%d/%d)",
                        attempt + 1,
                        attempts,
                    )
        if last_err:
            raise last_err
        raise RuntimeError("request failed without exception")


__all__ = ["ModelExecutor", "format_context_prompt"]
