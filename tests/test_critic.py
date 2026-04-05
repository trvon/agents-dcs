from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import dcs.critic as critic_mod
from dcs.critic import SelfCritic
from dcs.types import ContextBlock, ExecutionResult, ModelConfig


def test_parse_critique_accepts_nested_aliases_and_percentages() -> None:
    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    context = ContextBlock(
        content="ctx",
        sources=["src/search/search_engine.cpp", "include/yams/search/search_engine.h"],
        chunk_ids=["chunk-1", "chunk-2"],
        token_count=100,
        budget=200,
        utilization=0.5,
        chunks_included=2,
        chunks_considered=2,
    )
    text = """```json
        {
          "evaluation": {
            "context_usage": "75%",
            "missing": "constructor details\\nscoring path",
            "irrelevant": ["search_engine.cpp", "chunk-2"],
            "overall_score": "0.8",
            "queries": "SearchEngine constructor; fusion scoring",
        "notes": "Mostly grounded"
      }
    }
    ```"""

    parsed = critic._parse_critique(text, context)
    assert parsed is not None
    assert parsed.context_utilization == 0.75
    assert parsed.quality_score == 0.8
    assert parsed.missing_info == ["constructor details", "scoring path"]
    assert parsed.irrelevant_chunks == ["chunk-1", "chunk-2"]
    assert parsed.suggested_queries == ["SearchEngine constructor", "fusion scoring"]
    assert parsed.reasoning == "Mostly grounded"


@pytest.mark.asyncio
async def test_critique_retries_without_json_schema_on_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBadRequestError(Exception):
        pass

    calls: list[dict] = []

    class _Resp:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"context_utilization":0.5,"missing_info":[],"irrelevant_chunks":[],"quality_score":0.7,"suggested_queries":[],"reasoning":"ok"}'
                        }
                    }
                ]
            }

    class _Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise FakeBadRequestError("response_format.type must be json_schema or text")
            return _Resp()

    class _Chat:
        completions = _Completions()

    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b", max_output_tokens=1024))
    critic.client = SimpleNamespace(chat=_Chat())
    monkeypatch.setattr(critic_mod, "BadRequestError", FakeBadRequestError)

    parsed = await critic.critique(
        task="Explain search engine",
        context=ContextBlock(content="ctx", sources=[], chunk_ids=[]),
        result=ExecutionResult(output="answer", model="openai/gpt-oss-20b"),
    )

    assert parsed.quality_score == 0.7
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_critique_extracts_text_from_content_parts() -> None:
    class _Resp:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "output_text", "text": '{"context_utilization":0.4,'},
                                {
                                    "type": "output_text",
                                    "text": '"missing_info":[],"irrelevant_chunks":[],"quality_score":0.6,"suggested_queries":[],"reasoning":"fine"}',
                                },
                            ]
                        }
                    }
                ]
            }

    class _Completions:
        async def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    critic.client = SimpleNamespace(chat=_Chat())
    parsed = await critic.critique(
        task="Explain graph",
        context=ContextBlock(content="ctx", sources=[], chunk_ids=[]),
        result=ExecutionResult(output="answer", model="openai/gpt-oss-20b"),
    )

    assert parsed.quality_score == 0.6
    assert parsed.reasoning == "fine"


@pytest.mark.asyncio
async def test_qwen_critique_retries_when_json_schema_returns_empty() -> None:
    calls: list[dict] = []

    class _Resp:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self):
            return self.payload

    class _Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                return _Resp({"choices": [{"message": {"content": ""}}]})
            return _Resp(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"context_utilization":0.5,"missing_info":[],"irrelevant_chunks":[],"quality_score":0.65,"suggested_queries":[],"reasoning":"ok"}'
                            }
                        }
                    ]
                }
            )

    class _Chat:
        completions = _Completions()

    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    critic.client = SimpleNamespace(chat=_Chat())
    parsed = await critic.critique(
        task="Explain graph",
        context=ContextBlock(content="ctx", sources=[], chunk_ids=[]),
        result=ExecutionResult(output="answer", model="openai/gpt-oss-20b"),
    )

    assert parsed.quality_score == 0.65
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_critique_uses_reasoning_content_when_content_empty() -> None:
    class _Resp:
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"context_utilization":0.55,"missing_info":[],"irrelevant_chunks":[],"quality_score":0.72,"suggested_queries":[],"reasoning":"from reasoning"}',
                        }
                    }
                ]
            }

    class _Completions:
        async def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    critic.client = SimpleNamespace(chat=_Chat())
    parsed = await critic.critique(
        task="Explain graph",
        context=ContextBlock(content="ctx", sources=[], chunk_ids=[]),
        result=ExecutionResult(output="answer", model="openai/gpt-oss-20b"),
    )

    assert parsed.quality_score == 0.72
    assert parsed.reasoning == "from reasoning"


def test_qwen_critic_prompt_adds_no_think_and_strict_json_rules() -> None:
    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    messages = critic._build_critique_prompt(
        task="Explain graph",
        context=ContextBlock(content="ctx", sources=[], chunk_ids=[]),
        result=ExecutionResult(output="answer", model="openai/gpt-oss-20b"),
    )

    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "Do not emit <think> blocks." in system
    assert "/no_think" in system
    assert "Start immediately with '{'." in user


def test_critic_debug_artifact_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    critic = SelfCritic(ModelConfig(name="qwen_qwen3.5-122b-a10b"))
    monkeypatch.setenv("DCS_CRITIC_DEBUG_DIR", str(tmp_path))
    critic._dump_debug_artifact(
        task="Explain graph",
        messages=[{"role": "system", "content": "hi"}],
        content="raw output",
        note="unparseable_critique",
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    payload = files[0].read_text(encoding="utf-8")
    assert "unparseable_critique" in payload
    assert "raw output" in payload
