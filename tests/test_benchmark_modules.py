from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import coverage_benchmark as cov_bench
from benchmarks import report_benchmark as report_bench
from benchmarks import retrieval_benchmark as ret_bench
from dcs.types import (
    ContextBlock,
    Critique,
    EvalTask,
    IterationRecord,
    ModelConfig,
    PipelineConfig,
    PipelineResult,
    QuerySpec,
    QueryType,
    TaskType,
    YAMSChunk,
    YAMSQueryResult,
)


def _mk_task(task_id: str = "t1") -> EvalTask:
    return EvalTask(
        id=task_id,
        task_type=TaskType.QA,
        description="Describe mcp tools",
        ground_truth={"files": ["src/mcp/mcp_server.cpp"], "patterns": ["registerTool"]},
        evaluation={"quality_threshold": 0.7},
        tags=["qa", "mcp"],
    )


def test_coverage_benchmark_model_loading_and_checkpoint_io(tmp_path: Path) -> None:
    cfg_file = tmp_path / "models.yaml"
    cfg_file.write_text(
        """
backends:
  lm:
    base_url: http://localhost:1234/v1
    api_key: token
models:
  thinking:
    name: qwen-thinking
    backend: lm
""".strip(),
        encoding="utf-8",
    )

    cfg = cov_bench._load_models_config(cfg_file)
    mc = cov_bench._build_model_config(cfg, "thinking")
    assert mc.base_url == "http://localhost:1234/v1"
    assert mc.system_suffix == "/no_think"

    payload = {"k": [{"task_id": "x", "passed": True}]}
    ckpt = tmp_path / "ckpt.json"
    cov_bench._write_checkpoint(ckpt, payload)
    loaded = cov_bench._load_checkpoint(ckpt)
    assert loaded == payload

    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    assert cov_bench._load_checkpoint(tmp_path / "bad.json") == {}


def test_coverage_collect_sources_and_serialize() -> None:
    chunk = YAMSChunk(chunk_id="c1", content="x", source="src/a.cpp", score=0.9)
    q = YAMSQueryResult(
        spec=QuerySpec(query="q", query_type=QueryType.GREP, importance=1.0),
        chunks=[chunk],
    )
    it = IterationRecord(
        iteration=1,
        query_results=[q],
        context=ContextBlock(content="ctx", sources=["src/a.cpp"], chunks_included=1),
        critique=Critique(context_utilization=0.8, quality_score=0.9),
    )
    pr = PipelineResult(task="t", iterations=[it], total_latency_ms=10.0)
    srcs = cov_bench._collect_sources(pr)
    assert srcs == ["src/a.cpp"]

    result = cov_bench._serialize_results(
        [
            cov_bench.EvalResult(
                task_id="t",
                pipeline_result=pr,
                metrics={"quality_score": 1.0},
                passed=True,
            )
        ]
    )
    assert result[0]["pipeline"]["iterations"] == 1
    assert result[0]["sources"] == ["src/a.cpp"]


def test_coverage_preload_configs_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cov_bench, "preload_model", lambda name, **kwargs: calls.append(name) or True
    )
    monkeypatch.setattr(cov_bench, "get_context_length", lambda name: 32768)

    c1 = ModelConfig(name="m1", base_url="u", api_key="k")
    c2 = ModelConfig(name="m1", base_url="u", api_key="k")
    c3 = ModelConfig(name="m2", base_url="u", api_key="k")

    console = SimpleNamespace(print=lambda *args, **kwargs: None)
    cov_bench._preload_configs(console, [c1, c2, c3], retries=2, retry_backoff_s=1.0)

    assert calls == ["m1", "m2"]
    assert c1.context_window == 32768


def test_coverage_disables_heavy_secondary_calls() -> None:
    cfg = cov_bench._disable_heavy_secondary_calls(PipelineConfig())
    assert cfg.use_dspy_faithfulness is False
    assert cfg.no_ground_truth_mode is False


@pytest.mark.asyncio
async def test_coverage_run_suite_with_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeRunner:
        def __init__(self, config, task_dir):
            self.config = config

        def _decide_pass(self, task, metrics):
            return float(metrics.get("quality_score", 0.0)) >= 0.5

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        async def run(self, task: str) -> PipelineResult:
            it = IterationRecord(
                iteration=1,
                context=ContextBlock(content="ctx", sources=["src/a.cpp"], chunks_included=1),
                critique=Critique(context_utilization=1.0, quality_score=0.9),
            )
            return PipelineResult(task=task, iterations=[it], total_latency_ms=12.0)

    monkeypatch.setattr(cov_bench, "EvalRunner", FakeRunner)
    monkeypatch.setattr(cov_bench, "DCSPipeline", FakePipeline)
    monkeypatch.setattr(cov_bench, "evaluate_task", lambda task, pr: {"quality_score": 0.9})

    ckpt_data: dict[str, list[dict[str, object]]] = {}
    ckpt_path = tmp_path / "checkpoint.json"
    task = _mk_task("task-1")
    results = await cov_bench._run_suite(
        PipelineConfig(),
        [task],
        checkpoint=ckpt_data,
        checkpoint_key="k",
        checkpoint_path=ckpt_path,
    )

    assert len(results) == 1
    assert results[0].passed
    saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert "k" in saved and saved["k"][0]["task_id"] == "task-1"


@pytest.mark.asyncio
async def test_retrieval_run_task_heuristic_and_model_modes() -> None:
    task = _mk_task("r1")

    class FakePlanner:
        async def execute(self, specs, allow_adaptive=True):
            chunk = YAMSChunk(
                chunk_id="c1",
                content="registerTool",
                source="src/mcp/mcp_server.cpp",
                score=0.9,
            )
            return [
                YAMSQueryResult(
                    spec=QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=1.0),
                    chunks=[chunk],
                )
            ]

    class FakeDecomposer:
        async def decompose(self, task, max_queries, use_task_seeding=True):
            return [QuerySpec(query="q", query_type=QueryType.GREP, importance=1.0)]

    row_h = await ret_bench._run_task(
        task,
        decomposer=None,
        planner=FakePlanner(),
        max_queries=3,
        decompose_mode="heuristic",
        use_task_seeding=False,
    )
    row_m = await ret_bench._run_task(
        task,
        decomposer=FakeDecomposer(),
        planner=FakePlanner(),
        max_queries=3,
        decompose_mode="model",
        use_task_seeding=True,
    )

    assert row_h["metrics"]["file_hit_at_1"] == 1.0
    assert row_m["metrics"]["symbol_hit_at_1"] == 1.0


def test_retrieval_helpers_and_summary() -> None:
    assert ret_bench._task_type_from_str("qa") == TaskType.QA
    assert ret_bench._task_type_from_str("nope") is None
    assert ret_bench._match_file("/x/src/mcp/mcp_server.cpp", "src/mcp/mcp_server.cpp")
    assert ret_bench._is_noise_source("/repo/docs/readme.md")
    assert ret_bench._pattern_match("registerTool", "re:register.*")

    rows = [
        {
            "metrics": {
                "file_hit_at_1": 1.0,
                "file_hit_at_3": 1.0,
                "file_hit_at_5": 1.0,
                "file_recall_at_5": 1.0,
                "file_mrr": 1.0,
                "symbol_hit_at_5": 1.0,
                "grep_noise_rate": 0.0,
                "graph_useful_neighbor_ratio": 1.0,
                "retrieval_latency_ms": 10.0,
            }
        }
    ]
    s = ret_bench._summary(rows)
    assert s["file_hit_at_1"] == 1.0
    assert s["retrieval_latency_ms"] == 10.0


def test_report_benchmark_summarizes_and_compares() -> None:
    rows = [
        {
            "model": "gpt-oss-120b-executor",
            "task_id": "t1",
            "passed": True,
            "metrics": {"quality_score": 0.9, "total_latency_ms": 100.0},
        },
        {
            "model": "qwen-122b-executor",
            "task_id": "t1",
            "passed": False,
            "metrics": {"quality_score": 0.7, "total_latency_ms": 80.0},
        },
    ]
    summary = report_bench.summarize_models(rows)
    comparison = report_bench.compare_models(rows)
    assert summary["gpt-oss-120b-executor"]["pass_rate_mean"] == 1.0
    assert comparison["wins"]["gpt-oss-120b-executor"] == 1
