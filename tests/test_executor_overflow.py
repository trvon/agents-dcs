from __future__ import annotations

from dcs.executor import ModelExecutor, _is_model_unloaded_error, _parse_context_overflow
from dcs.types import ModelConfig


def test_parse_context_overflow_with_numbers() -> None:
    msg = (
        "Error code: 400 - {'error': 'The number of tokens to keep from the initial "
        "prompt (9800) is greater than the context length (8192).'}"
    )
    keep, ctx = _parse_context_overflow(msg)
    assert keep == 9800
    assert ctx == 8192


def test_parse_context_overflow_without_numbers() -> None:
    msg = "tokens to keep from the initial prompt is greater than the context length"
    keep, ctx = _parse_context_overflow(msg)
    assert keep is None
    assert ctx is None


def test_is_model_unloaded_error_matches_known_messages() -> None:
    assert _is_model_unloaded_error("Error code: 400 - {'error': 'Model is unloaded.'}")
    assert _is_model_unloaded_error('Failed to load model "qwen3.5-35b-a3b"')


def test_is_model_unloaded_error_ignores_other_errors() -> None:
    assert not _is_model_unloaded_error(
        "Error code: 400 - {'error': 'tokens to keep from the initial prompt is greater than context length'}"
    )


def test_model_backoff_grows_with_attempts() -> None:
    ex = ModelExecutor.__new__(ModelExecutor)
    ex.config = ModelConfig(name="demo", retry_backoff_s=2.0)
    first = ex._compute_retry_backoff(0)
    second = ex._compute_retry_backoff(1)
    assert first == 2.0
    assert second == 4.0
