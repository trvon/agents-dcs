from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

try:
    import lmstudio as lms  # type: ignore
except Exception:  # pragma: no cover
    lms = None  # type: ignore


_MODEL_CACHE: dict[str, Any] = {}


def is_available() -> bool:
    return lms is not None


def _get_model(
    model_name: str,
    *,
    context_length: int | None = None,
    keep_model_in_memory: bool = True,
):
    key = model_name or "__default__"
    if context_length and context_length > 0:
        key = f"{key}::ctx={int(context_length)}::keep={1 if keep_model_in_memory else 0}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if lms is None:
        return None
    try:
        load_cfg_dict: dict[str, Any] = {}
        if keep_model_in_memory:
            load_cfg_dict["keep_model_in_memory"] = True
        if context_length and context_length > 0:
            load_cfg_dict["context_length"] = int(context_length)

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
    keep_model_in_memory: bool = True,
    retries: int = 3,
    retry_backoff_s: float = 2.0,
    max_tokens: int = 8,
    ready_timeout_s: float = 600.0,
    ready_poll_s: float = 2.0,
) -> bool:
    """Best-effort model preload + warmup through LM Studio.

    This helps keep large fallback models resident before long benchmark sweeps.
    """
    if not model_name.strip():
        return False

    # Native SDK pin (if available).
    _get_model(
        model_name,
        context_length=context_length,
        keep_model_in_memory=keep_model_in_memory,
    )

    # OpenAI-compatible warmup call to ensure the model is actually loaded.
    client = OpenAI(base_url=base_url, api_key=api_key)
    last_error: Exception | None = None
    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return exactly: ok"},
                    {"role": "user", "content": "ok"},
                ],
                temperature=0.0,
                max_tokens=max(1, int(max_tokens)),
            )
            logger.info("lmstudio: model warmup ok: %s", model_name)
            return True
        except Exception as e:  # pragma: no cover
            last_error = e

            elapsed = time.monotonic() - start
            max_attempts = max(1, int(retries))
            if attempts >= max_attempts and elapsed >= max(1.0, float(ready_timeout_s)):
                break

            sleep_for = max(
                float(ready_poll_s),
                max(0.0, float(retry_backoff_s)) * min(attempts, max_attempts),
            )
            time.sleep(sleep_for)

    logger.warning("lmstudio: model warmup failed: %s (%s)", model_name, last_error)
    return False


def preload_models(
    model_names: list[str],
    *,
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    retries: int = 3,
    retry_backoff_s: float = 2.0,
    context_length: int | None = None,
    keep_model_in_memory: bool = True,
    ready_timeout_s: float = 600.0,
    ready_poll_s: float = 2.0,
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
            keep_model_in_memory=keep_model_in_memory,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            ready_timeout_s=ready_timeout_s,
            ready_poll_s=ready_poll_s,
        )
    return status
