from __future__ import annotations

from dcs import lmstudio_context as ctx


def test_effective_load_context_length_respects_minimum() -> None:
    assert ctx._effective_load_context_length(4096, min_ready_context_length=65535) == 65535
    assert ctx._effective_load_context_length(70000, min_ready_context_length=65535) == 70000
    assert ctx._effective_load_context_length(None, min_ready_context_length=65535) == 65535


def test_rest_api_base_url_removes_v1_suffix() -> None:
    assert (
        ctx._rest_api_base_url_from_openai_base("http://localhost:1234/v1")
        == "http://localhost:1234/api/v1"
    )
    assert (
        ctx._rest_api_base_url_from_openai_base("http://localhost:1234")
        == "http://localhost:1234/api/v1"
    )


def test_extract_loaded_context_length_prefers_load_config() -> None:
    assert ctx._extract_loaded_context_length({"load_config": {"context_length": 65535}}) == 65535
    assert ctx._extract_loaded_context_length({"context_length": 32768}) == 32768
    assert ctx._extract_loaded_context_length({}) is None
