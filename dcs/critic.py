"""Reflexion-style self-critique for DCS.

The critic evaluates whether the assembled context actually helped produce the
model output, identifies missing information, and flags irrelevant chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import (  # type: ignore[import-not-found]
        APIConnectionError,
        APITimeoutError,
        AsyncOpenAI,
        BadRequestError,
    )
except Exception:  # pragma: no cover
    # Allow importing this module in environments without the OpenAI SDK.
    APIConnectionError = Exception  # type: ignore[assignment]
    APITimeoutError = Exception  # type: ignore[assignment]
    BadRequestError = Exception  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]

from dcs.runtime_config import load_runtime_settings
from dcs.types import ContextBlock, Critique, ExecutionResult, ModelConfig

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object from a text blob.

    Small models often return JSON plus extra prose. We find the first balanced
    {...} region while respecting quoted strings. Thinking-mode models may wrap
    output in <think>...</think> blocks; strip those first.
    """

    if not text:
        return None

    # Strip <think>...</think> blocks (qwen3, deepseek-r1, etc.)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text  # fallback to original if stripping ate everything

    # Also try stripping markdown code fences around JSON
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        try:
            import json as _json

            _json.loads(fence_match.group(1))
            return fence_match.group(1)
        except Exception:
            pass

    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]

    return None


def _extract_first_json_array(text: str) -> str | None:
    """Extract the first top-level JSON array from a text blob."""
    if not text:
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text

    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        try:
            import json as _json

            _json.loads(fence_match.group(1))
            return fence_match.group(1)
        except Exception:
            pass

    start = cleaned.find("[")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]

    return None


def _try_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    # Lenient cleanup: strip comments, remove trailing commas, normalize quotes.
    cleaned = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # If there are no double quotes but many single quotes, convert to double quotes.
    if cleaned.count('"') == 0 and cleaned.count("'") >= 2:
        cleaned = cleaned.replace("'", '"')

    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [p.strip(" -*\t\n\r") for p in re.split(r"[\n;,]+", text) if p.strip()]
        return [p for p in parts if p]
    return []


def _as_score(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                return float(text) / 100.0
            except Exception:
                return 0.0
        try:
            return float(text)
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


class SelfCritic:
    """Ask a (possibly larger) model to critique context and output."""

    def __init__(self, config: ModelConfig):
        self.config = config
        if AsyncOpenAI is None:  # pragma: no cover
            self.client = None
        else:
            self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    @staticmethod
    def _critique_json_schema() -> dict[str, Any]:
        return {
            "name": "dcs_critique",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "context_utilization": {"type": "number"},
                    "missing_info": {"type": "array", "items": {"type": "string"}},
                    "irrelevant_chunks": {"type": "array", "items": {"type": "string"}},
                    "quality_score": {"type": "number"},
                    "suggested_queries": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "context_utilization",
                    "missing_info",
                    "irrelevant_chunks",
                    "quality_score",
                    "suggested_queries",
                    "reasoning",
                ],
            },
        }

    def _supports_json_schema(self) -> bool:
        name = str(self.config.name or "").lower()
        return any(k in name for k in ("qwen", "gpt-oss", "openai/"))

    def _is_qwen_model(self) -> bool:
        return "qwen" in str(self.config.name or "").lower()

    def _critic_system_suffix(self) -> str:
        suffix = (self.config.system_suffix or "").strip()
        if suffix:
            return suffix
        if self._is_qwen_model():
            return "/no_think"
        return ""

    def _dump_debug_artifact(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        content: str,
        note: str,
        raw: dict[str, Any] | None = None,
    ) -> None:
        debug_dir = os.environ.get("DCS_CRITIC_DEBUG_DIR", "").strip()
        if not debug_dir:
            runtime = load_runtime_settings(Path(__file__).resolve().parents[1])
            debug_dir = str(runtime.critic_debug_dir or "").strip()
        if not debug_dir:
            return
        try:
            path = Path(debug_dir)
            path.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9]+", "-", (task or "critic").lower()).strip("-") or "critic"
            stamp = str(int(time.time() * 1000))
            out = path / f"{stamp}-{slug}.json"
            out.write_text(
                json.dumps(
                    {
                        "model": self.config.name,
                        "note": note,
                        "task": task,
                        "messages": messages,
                        "content": content,
                        "raw": raw,
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to write critic debug artifact", exc_info=True)

    def _extract_message_content(self, raw: dict[str, Any]) -> str:
        choices = raw.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
        reasoning_content = msg.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            return "\n".join(parts)
        return ""

    async def _request_critique(
        self, messages: list[dict[str, Any]], *, use_json_schema: bool
    ) -> tuple[str, float, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(min(self.config.max_output_tokens, 768)),
        }
        if use_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": self._critique_json_schema(),
            }
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        raw = resp.model_dump()
        return self._extract_message_content(raw), latency_ms, raw

    async def critique(self, task: str, context: ContextBlock, result: ExecutionResult) -> Critique:
        if self.client is None:
            c = self._heuristic_critique(task=task, context=context, result=result)
            c.reasoning = (
                "OpenAI SDK not available; using heuristic critique. " + (c.reasoning or "")
            ).strip()
            return c

        messages = self._build_critique_prompt(task=task, context=context, result=result)

        try:
            try:
                content, latency_ms, raw = await self._request_critique(
                    messages,
                    use_json_schema=self._supports_json_schema(),
                )
                if self._supports_json_schema() and not content.strip():
                    self._dump_debug_artifact(
                        task=task,
                        messages=messages,
                        content=content,
                        note="empty_json_schema_critique",
                        raw=raw,
                    )
                    content, latency_ms, raw = await self._request_critique(
                        messages, use_json_schema=False
                    )
            except BadRequestError as e:
                if "response_format" not in str(e).lower():
                    raise
                content, latency_ms, raw = await self._request_critique(
                    messages, use_json_schema=False
                )
            logger.debug("Critique call completed in %.1fms", latency_ms)

            parsed = self._parse_critique(content, context=context)
            if parsed is not None:
                return parsed

            self._dump_debug_artifact(
                task=task,
                messages=messages,
                content=content,
                note="unparseable_critique",
                raw=raw,
            )
            logger.warning("Critique model output unparseable; falling back to heuristics")
            c = self._heuristic_critique(task=task, context=context, result=result)
            c.reasoning = (
                "Model critique response was unparseable; using heuristic critique. "
                + (c.reasoning or "")
            ).strip()
            return c

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning("Critique model API unavailable; using heuristics: %s", e, exc_info=True)
            c = self._heuristic_critique(task=task, context=context, result=result)
            c.reasoning = (
                f"Critique model call failed ({type(e).__name__}); using heuristic critique. "
                + (c.reasoning or "")
            ).strip()
            return c
        except BadRequestError as e:
            logger.warning("Critique prompt too large for model context; using heuristics: %s", e)
            c = self._heuristic_critique(task=task, context=context, result=result)
            c.reasoning = (
                "Critique prompt exceeded model context window; using heuristic critique. "
                + (c.reasoning or "")
            ).strip()
            return c
        except Exception as e:
            self._dump_debug_artifact(
                task=task,
                messages=messages,
                content=str(e),
                note="critic_exception",
            )
            logger.exception("Unexpected critique failure; using heuristics")
            c = self._heuristic_critique(task=task, context=context, result=result)
            c.reasoning = (
                f"Critique model call failed ({type(e).__name__}: {e}); using heuristic critique. "
                + (c.reasoning or "")
            ).strip()
            return c

    def _truncate_for_critic(self, text: str, max_chars: int) -> str:
        """Truncate text to fit within critic's context budget.

        The critic doesn't need the full context/output verbatim — a representative
        sample is enough for quality evaluation.
        """
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return (
            text[:half]
            + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n"
            + text[-half:]
        )

    def _build_critique_prompt(
        self, task: str, context: ContextBlock, result: ExecutionResult
    ) -> list[dict[str, Any]]:
        schema = (
            "Return a JSON object with this exact schema:\n\n"
            "{\n"
            '  "context_utilization": number,\n'
            '  "missing_info": string[],\n'
            '  "irrelevant_chunks": string[],\n'
            '  "quality_score": number,\n'
            '  "suggested_queries": string[],\n'
            '  "reasoning": string\n'
            "}\n\n"
            "Rules:\n"
            "- Output the JSON object FIRST and ONLY ONCE.\n"
            "- Use conservative scores when uncertain.\n"
            "- irrelevant_chunks must be a subset of the provided chunk_ids.\n"
            "- Use decimal scores between 0.0 and 1.0, not percentages.\n"
        )

        sys = (
            "You are a strict evaluator for a Dynamic Context Scaffold (DCS). "
            "Judge whether the retrieved context helped the model produce a good output. "
            "Return ONLY the JSON object.\n\n" + schema
        )
        if self._is_qwen_model():
            sys += (
                "\n\nQwen-specific rules:\n"
                "- Do not emit <think> blocks.\n"
                "- Do not use markdown fences.\n"
                "- Start with '{' and end with '}'.\n"
            )
        suffix = self._critic_system_suffix()
        if suffix:
            sys += "\n" + suffix

        sources = context.sources or []
        chunk_ids = context.chunk_ids or []
        context_meta = (
            f"tokens={context.token_count} budget={context.budget} "
            f"util={context.utilization:.2f} chunks={context.chunks_included}/{context.chunks_considered}"
        )

        # Aggressive budget: critic prompt must fit in ~50% of context window to leave
        # room for the response (up to 512 tokens) and model overhead. Estimate ~3.5
        # chars per token for English mixed with code.
        chars_per_token = 3.5
        response_reserve = 600  # tokens reserved for response
        model_overhead = 200  # tokens for model internal overhead / special tokens
        sys_tokens_est = int(len(sys) / chars_per_token) + 50  # +overhead
        available_tokens = max(
            200,
            int(self.config.context_window) - response_reserve - model_overhead - sys_tokens_est,
        )
        # Use only 60% of available space to be safe with token estimation errors
        available_chars = int(available_tokens * chars_per_token * 0.6)

        # Split available space: 40% context, 40% output, 20% task+meta
        meta_budget = int(available_chars * 0.20)
        content_budget = int(available_chars * 0.40)
        output_budget = int(available_chars * 0.40)

        ctx_content = self._truncate_for_critic((context.content or "").strip(), content_budget)
        model_output = self._truncate_for_critic((result.output or "").strip(), output_budget)
        task_text = self._truncate_for_critic((task or "").strip(), meta_budget)

        user = (
            f"TASK: {task_text}\n\n"
            f"CONTEXT ({context_meta}): sources={json.dumps(sources[:10])}, chunk_ids={json.dumps(chunk_ids[:10])}\n"
            f"{ctx_content}\n\n"
            f"OUTPUT:\n{model_output}\n\n"
            "Return the JSON object only. Start immediately with '{'."
        )

        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    def _parse_critique(self, text: str, context: ContextBlock) -> Critique | None:
        blob = _extract_first_json_object(text) or ""
        data: dict[str, Any] | list[Any] | None = None
        if blob:
            data = _try_parse_json(blob)

        if data is None:
            arr = _extract_first_json_array(text)
            if arr:
                data = _try_parse_json(arr)

        if data is None:
            data = _try_parse_json(text)

        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict)), None)

        if not isinstance(data, dict):
            return None

        for nested_key in ("critique", "evaluation", "result"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                data = nested
                break

        missing_info = _as_string_list(
            data.get("missing_info", data.get("missing", data.get("gaps", [])))
        )

        irrelevant_chunks = _as_string_list(
            data.get(
                "irrelevant_chunks",
                data.get("irrelevant", data.get("irrelevant_chunk_ids", [])),
            )
        )

        # Enforce subset of provided chunk IDs (when present) to keep downstream logic stable.
        known = set(context.chunk_ids or [])
        if known:
            alias_map: dict[str, str] = {}
            for cid in known:
                alias_map[cid] = cid
                alias_map[cid.lower()] = cid
            for src, cid in zip(context.sources or [], context.chunk_ids or []):
                if not src or not cid:
                    continue
                alias_map[src] = cid
                alias_map[src.lower()] = cid
                base = src.rsplit("/", 1)[-1]
                if base:
                    alias_map[base] = cid
                    alias_map[base.lower()] = cid
            normalized: list[str] = []
            for item in irrelevant_chunks:
                key = item if item in alias_map else item.lower()
                cid = alias_map.get(key)
                if cid and cid not in normalized:
                    normalized.append(cid)
            irrelevant_chunks = normalized

        suggested_queries = _as_string_list(
            data.get("suggested_queries", data.get("queries", data.get("followup_queries", [])))
        )

        reasoning = data.get("reasoning", data.get("explanation", data.get("notes", "")))
        if not isinstance(reasoning, str):
            reasoning = ""

        # Some models use alternate key names; tolerate a few common variants.
        context_util = _as_score(
            data.get(
                "context_utilization", data.get("context_usage", data.get("context_score", 0.0))
            )
        )
        quality = _as_score(
            data.get("quality_score", data.get("quality", data.get("overall_score", 0.0)))
        )

        return Critique(
            context_utilization=_clamp01(context_util),
            missing_info=missing_info,
            irrelevant_chunks=irrelevant_chunks,
            quality_score=_clamp01(quality),
            suggested_queries=suggested_queries,
            reasoning=reasoning.strip(),
        )

    def _heuristic_critique(
        self, task: str, context: ContextBlock, result: ExecutionResult
    ) -> Critique:
        output = (result.output or "").strip()
        sources = context.sources or []
        chunk_ids = context.chunk_ids or []

        used_hits = 0
        for s in sources:
            if not s:
                continue
            if s in output:
                used_hits += 1
                continue
            base = s.rsplit("/", 1)[-1]
            if base and base in output:
                used_hits += 1

        for cid in chunk_ids:
            if cid and cid in output:
                used_hits += 1

        denom = max(1, len(set(sources)) + len(set(chunk_ids)))
        utilization_est = _clamp01(used_hits / denom)

        # Conservative quality heuristic.
        out_len = len(output)
        quality = 0.35
        if out_len > 200:
            quality += 0.15
        if out_len > 800:
            quality += 0.10

        lowered = output.lower()
        if "error:" in lowered or "traceback" in lowered:
            quality -= 0.25
        if "i don't know" in lowered or "insufficient" in lowered or "not enough" in lowered:
            quality -= 0.15
        quality = _clamp01(quality)

        missing: list[str] = []
        suggested: list[str] = []

        if not (context.content or "").strip():
            missing.append("No retrieved context was provided (context content is empty).")
        if utilization_est < 0.2 and (context.content or "").strip():
            missing.append(
                "Relevant citations/anchors (file paths, identifiers) to connect output to context."
            )

        # Basic query suggestions from the task text.
        task_text = (task or "").strip()
        if task_text:
            suggested.append(f"semantic: {task_text}")
            words = [w for w in re.split(r"\W+", task_text) if len(w) >= 5]
            if words:
                suggested.append(f"grep: {words[0]}")

        # Irrelevant chunks: we can't reliably decide without per-chunk content; be conservative.
        irrelevant: list[str] = []

        return Critique(
            context_utilization=utilization_est,
            missing_info=missing,
            irrelevant_chunks=irrelevant,
            quality_score=quality,
            suggested_queries=suggested,
            reasoning=(
                "Heuristic estimate based on whether output references provided sources/chunk IDs and "
                "basic output health signals."
            ),
        )
