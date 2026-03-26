from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import request as urllib_request

from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    import lmstudio as lms  # type: ignore
except Exception:  # pragma: no cover
    lms = None  # type: ignore


_MODEL_CACHE: dict[str, Any] = {}
DEFAULT_MIN_READY_CONTEXT_LENGTH = 65535


def is_available() -> bool:
    return lms is not None


def _effective_load_context_length(
    context_length: int | None,
    *,
    min_ready_context_length: int | None = DEFAULT_MIN_READY_CONTEXT_LENGTH,
) -> int | None:
    requested = int(context_length or 0)
    minimum = int(min_ready_context_length or 0)
    target = max(requested, minimum)
    return target if target > 0 else None


def _rest_api_base_url_from_openai_base(base_url: str) -> str:
    raw = str(base_url or "http://localhost:1234/v1").rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    return raw + "/api/v1"


def _extract_loaded_context_length(payload: dict[str, Any]) -> int | None:
    if not isinstance(payload, dict):
        return None
    load_cfg = payload.get("load_config")
    if isinstance(load_cfg, dict):
        try:
            value = load_cfg.get("context_length")
            if value is not None:
                return int(value)
        except Exception:
            pass
    try:
        value = payload.get("context_length")
        if value is not None:
            return int(value)
    except Exception:
        pass
    return None


def _load_model_via_rest(
    model_name: str,
    *,
    base_url: str,
    api_key: str,
    context_length: int | None,
    timeout_s: float,
) -> dict[str, Any]:
    url = _rest_api_base_url_from_openai_base(base_url) + "/models/load"
    payload: dict[str, Any] = {"model": model_name, "echo_load_config": True}
    if context_length and int(context_length) > 0:
        payload["context_length"] = int(context_length)

    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=max(1.0, float(timeout_s))) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw) if raw.strip() else {}
    return data if isinstance(data, dict) else {}


def _get_model(
    model_name: str,
    *,
    context_length: int | None = None,
    min_ready_context_length: int | None = DEFAULT_MIN_READY_CONTEXT_LENGTH,
    keep_model_in_memory: bool = True,
):
    effective_context_length = _effective_load_context_length(
        context_length,
        min_ready_context_length=min_ready_context_length,
    )
    key = model_name or "__default__"
    if effective_context_length and effective_context_length > 0:
        key = f"{key}::ctx={int(effective_context_length)}::keep={1 if keep_model_in_memory else 0}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if lms is None:
        return None
    try:
        load_cfg_dict: dict[str, Any] = {}
        if keep_model_in_memory:
            load_cfg_dict["keep_model_in_memory"] = True
        if effective_context_length and effective_context_length > 0:
            load_cfg_dict["context_length"] = int(effective_context_length)

        load_cfg: Any = None
        if load_cfg_dict:
            if hasattr(lms, "LlmLoadModelConfig"):
                try:
                    load_cfg = lms.LlmLoadModelConfig(**load_cfg_dict)
                except Exception:
                    load_cfg = load_cfg_dict
            else:
                load_cfg = load_cfg_dict

        model = lms.llm(model_name, config=load_cfg) if model_name else lms.llm(config=load_cfg)
        _MODEL_CACHE[key] = model
        return model
    except Exception as e:  # pragma: no cover
        logger.warning("lmstudio: failed to load model %r: %s", model_name, e)
        return None


def get_context_length(model_name: str) -> int | None:
    model = _get_model(model_name)
    if model is None:
        return None
    try:
        return int(model.get_context_length())
    except Exception as e:  # pragma: no cover
        logger.warning("lmstudio: get_context_length failed for %r: %s", model_name, e)
        return None


def count_prompt_tokens(model_name: str, messages: list[dict[str, Any]]) -> int | None:
    model = _get_model(model_name)
    if model is None or lms is None:
        return None
    try:
        try:
            chat = lms.Chat.from_history(messages)  # type: ignore[arg-type]
        except Exception:
            chat = lms.Chat.from_history({"messages": messages})  # type: ignore[arg-type]
        formatted = model.apply_prompt_template(chat)
        return int(len(model.tokenize(formatted)))
    except Exception as e:  # pragma: no cover
        logger.warning("lmstudio: tokenization failed for %r: %s", model_name, e)
        return None


def preload_model(
    model_name: str,
    *,
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    context_length: int | None = None,
    min_ready_context_length: int = DEFAULT_MIN_READY_CONTEXT_LENGTH,
    keep_model_in_memory: bool = True,
    retries: int = 3,
    retry_backoff_s: float = 2.0,
    max_tokens: int = 8,
    ready_timeout_s: float = 600.0,
    ready_poll_s: float = 2.0,
    required_successes: int = 1,
) -> bool:
    """Best-effort model preload + warmup through LM Studio.

    This helps keep large fallback models resident before long benchmark sweeps.
    """
    if not model_name.strip():
        return False

    target_context_length = _effective_load_context_length(
        context_length,
        min_ready_context_length=min_ready_context_length,
    )

    client = OpenAI(base_url=base_url, api_key=api_key)
    last_error: Exception | None = None
    max_attempts = max(1, int(retries))
    required_probes = max(1, int(required_successes))

    for attempt in range(1, max_attempts + 1):
        try:
            load_info = _load_model_via_rest(
                model_name,
                base_url=base_url,
                api_key=api_key,
                context_length=target_context_length,
                timeout_s=min(max(15.0, float(ready_timeout_s)), 300.0),
            )
            actual_context = _extract_loaded_context_length(load_info)
            if (
                target_context_length is not None
                and actual_context is not None
                and int(actual_context) < int(target_context_length)
            ):
                logger.warning(
                    "lmstudio: loaded context shorter than requested for %s (requested=%s actual=%s)",
                    model_name,
                    target_context_length,
                    actual_context,
                )

            for _ in range(required_probes):
                client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Return exactly: ok"},
                        {"role": "user", "content": "ok"},
                    ],
                    temperature=0.0,
                    max_tokens=max(1, int(max_tokens)),
                )

            logger.info(
                "lmstudio: model warmup ok: %s (context=%s, attempts=%d)",
                model_name,
                actual_context,
                attempt,
            )
            return True
        except Exception as e:  # pragma: no cover
            last_error = e
            if attempt >= max_attempts:
                break
            time.sleep(max(float(ready_poll_s), max(0.0, float(retry_backoff_s)) * attempt))

    logger.warning(
        "lmstudio: model warmup failed: %s (%s, attempts=%d)",
        model_name,
        last_error,
        max_attempts,
    )
    return False


def preload_models(
    model_names: list[str],
    *,
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    retries: int = 3,
    retry_backoff_s: float = 2.0,
    context_length: int | None = None,
    min_ready_context_length: int = DEFAULT_MIN_READY_CONTEXT_LENGTH,
    keep_model_in_memory: bool = True,
    ready_timeout_s: float = 600.0,
    ready_poll_s: float = 2.0,
    required_successes: int = 2,
) -> dict[str, bool]:
    status: dict[str, bool] = {}
    seen: set[str] = set()
    for name in model_names:
        n = (name or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        status[n] = preload_model(
            n,
            base_url=base_url,
            api_key=api_key,
            context_length=context_length,
            min_ready_context_length=min_ready_context_length,
            keep_model_in_memory=keep_model_in_memory,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            ready_timeout_s=ready_timeout_s,
            ready_poll_s=ready_poll_s,
            required_successes=required_successes,
        )
    return status
