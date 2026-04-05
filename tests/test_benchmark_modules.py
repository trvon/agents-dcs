from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import coverage_benchmark as cov_bench
from benchmarks import report_benchmark as report_bench
from benchmarks import retrieval_benchmark as ret_bench
from dcs import pipeline as pipe_mod
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

    raw = cov_bench._build_model_config_or_id(
        cfg, "qwen/qwen3-coder-next", preferred_role="executor", default_temperature=1.0
    )
    assert raw.name == "qwen/qwen3-coder-next"
    assert raw.temperature == 1.0
    assert cov_bench._is_lmstudio_backend("http://localhost:1234/v1") is True
    assert cov_bench._is_lmstudio_backend("http://127.0.0.1:8080/v1") is False

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


@pytest.mark.asyncio
async def test_coverage_maybe_attaches_plan_review(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = PipelineResult(task="t", final_output="done")
    task = EvalTask(
        id="t1",
        task_type=TaskType.CODING,
        description="Tests First\n1. Add regression test",
    )

    class FakeReviewer:
        def __init__(self, cfg):
            self.cfg = cfg

        async def review(self, review_input):
            return SimpleNamespace(coverage_score=0.8, executed_well=True, summary="ok")

    monkeypatch.setattr(cov_bench, "PlanReviewer", FakeReviewer)
    updated = await cov_bench._maybe_attach_plan_review(PipelineConfig(), task, pr, enabled=True)
    assert updated is pr
    assert updated.plan_review is not None
    assert updated.plan_review.coverage_score == 0.8


def test_coverage_serialize_includes_plan_review_summary() -> None:
    pr = PipelineResult(
        task="t",
        final_output="done",
        plan_review=SimpleNamespace(coverage_score=0.9, executed_well=True, summary="ok"),
    )
    result = cov_bench._serialize_results(
        [cov_bench.EvalResult(task_id="t", pipeline_result=pr, metrics={}, passed=True)]
    )
    assert result[0]["pipeline"]["plan_review"]["coverage_score"] == 0.9


def test_coverage_preload_configs_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cov_bench, "preload_model", lambda name, **kwargs: calls.append(name) or True
    )
    monkeypatch.setattr(cov_bench, "get_context_length", lambda name: 32768)

    c1 = ModelConfig(name="m1", base_url="http://localhost:1234/v1", api_key="k")
    c2 = ModelConfig(name="m1", base_url="http://localhost:1234/v1", api_key="k")
    c3 = ModelConfig(name="m2", base_url="http://localhost:1234/v1", api_key="k")

    console = SimpleNamespace(print=lambda *args, **kwargs: None)
    cov_bench._preload_configs(console, [c1, c2, c3], retries=2, retry_backoff_s=1.0)

    assert calls == ["m1", "m2"]
    assert c1.context_window == 32768


def test_coverage_disables_heavy_secondary_calls() -> None:
    cfg = cov_bench._disable_heavy_secondary_calls(PipelineConfig())
    assert cfg.use_dspy_faithfulness is False
    assert cfg.no_ground_truth_mode is False


def test_pipeline_config_allows_codemap_disable() -> None:
    cfg = PipelineConfig(codemap_budget=0)
    assert cfg.codemap_budget == 0


def test_pipeline_config_supports_conditional_codemap_caps() -> None:
    cfg = PipelineConfig(codemap_max_files=5, codemap_max_symbols_per_file=8)
    assert cfg.codemap_max_files == 5
    assert cfg.codemap_max_symbols_per_file == 8


def test_pipeline_config_supports_dspy_retrieval_rerank() -> None:
    cfg = PipelineConfig(
        use_dspy_retrieval_rerank=True,
        dspy_retrieval_top_k=5,
        dspy_retrieval_max_tokens=16384,
        dspy_retrieval_demo_count=4,
        dspy_retrieval_prefer_json=True,
    )
    assert cfg.use_dspy_retrieval_rerank is True
    assert cfg.dspy_retrieval_top_k == 5
    assert cfg.dspy_retrieval_max_tokens == 16384
    assert cfg.dspy_retrieval_demo_count == 4
    assert cfg.dspy_retrieval_prefer_json is True
    assert pipe_mod._is_lmstudio_backend("http://localhost:1234/v1") is True
    assert pipe_mod._is_lmstudio_backend("http://127.0.0.1:8080/v1") is False


def test_retrieval_model_resolution_prefers_matching_role_and_raw_ids() -> None:
    models_cfg = {
        "backends": {"lmstudio": {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"}},
        "models": {
            "gpt-oss-20b": {
                "backend": "lmstudio",
                "name": "openai/gpt-oss-20b",
                "temperature": 1.0,
                "role": "executor",
            },
            "gpt-oss-20b-critic": {
                "backend": "lmstudio",
                "name": "openai/gpt-oss-20b",
                "temperature": 0.0,
                "role": "critic",
            },
        },
    }

    exec_cfg = ret_bench._build_model_config_or_id(
        models_cfg,
        "openai/gpt-oss-20b",
        preferred_role="executor",
        default_temperature=1.0,
    )
    critic_cfg = ret_bench._build_model_config_or_id(
        models_cfg,
        "openai/gpt-oss-20b",
        preferred_role="critic",
        default_temperature=0.0,
    )

    assert exec_cfg.name == "openai/gpt-oss-20b"
    assert critic_cfg.name == "openai/gpt-oss-20b"
    assert exec_cfg.temperature == 1.0
    assert critic_cfg.temperature == 0.0
    assert exec_cfg.base_url == "http://localhost:1234/v1"


def test_retrieval_builds_mixed_suite_dspy_demos() -> None:
    tasks = [
        EvalTask(
            id="coding-a",
            task_type=TaskType.CODING,
            description="Explain SearchEngine hybrid search",
            ground_truth={
                "files": ["src/search/search_engine.cpp", "include/yams/search/search_engine.h"],
                "symbols": ["SearchEngine", "hybrid"],
            },
        ),
        EvalTask(
            id="qa-b",
            task_type=TaskType.QA,
            description="Describe daemon architecture",
            ground_truth={
                "files": ["include/yams/daemon/components/ServiceManager.h"],
                "symbols": ["ServiceManager", "InternalEventBus"],
            },
        ),
        EvalTask(
            id="coding-c",
            task_type=TaskType.CODING,
            description="Describe ResourceGovernor",
            ground_truth={
                "files": ["include/yams/daemon/components/ResourceGovernor.h"],
                "symbols": ["ResourceGovernor"],
            },
        ),
    ]

    demos = ret_bench._build_dspy_rerank_demos(tasks, current_task=tasks[0], limit=2)
    assert len(demos) == 2
    assert all("candidates_json" in demo for demo in demos)
    assert all(demo["query"] != tasks[0].description for demo in demos)
    assert any("ServiceManager" in demo["candidates_json"] for demo in demos)


def test_retrieval_builds_optimizer_examples_and_metric() -> None:
    tasks = [
        EvalTask(
            id="coding-a",
            task_type=TaskType.CODING,
            description="Explain SearchEngine hybrid search",
            ground_truth={
                "files": ["src/search/search_engine.cpp", "include/yams/search/search_engine.h"],
                "symbols": ["SearchEngine", "hybrid"],
            },
        ),
        EvalTask(
            id="qa-b",
            task_type=TaskType.QA,
            description="Describe daemon architecture",
            ground_truth={
                "files": ["include/yams/daemon/components/ServiceManager.h"],
                "symbols": ["ServiceManager", "InternalEventBus"],
            },
        ),
    ]

    examples = ret_bench._build_optimizer_rerank_examples(tasks, limit=2)
    assert len(examples) == 2
    assert all(hasattr(ex, "inputs") for ex in examples)

    good = SimpleNamespace(ranked_ids=[1, 2, 3])
    bad = SimpleNamespace(ranked_ids=[5, 4, 3])
    assert ret_bench._dspy_rerank_metric(examples[0], good) > ret_bench._dspy_rerank_metric(
        examples[0], bad
    )


def test_retrieval_counts_predictor_demos() -> None:
    class FakeProgram:
        def named_predictors(self):
            return [("a", SimpleNamespace(demos=[1, 2])), ("b", SimpleNamespace(demos=[3]))]

    assert ret_bench._count_predictor_demos(FakeProgram()) == 3


@pytest.mark.asyncio
async def test_retrieval_run_suite_builds_compiled_predictor_per_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        EvalTask(
            id="a",
            task_type=TaskType.CODING,
            description="A",
            ground_truth={"files": ["src/a.cpp"]},
        ),
        EvalTask(
            id="b", task_type=TaskType.QA, description="B", ground_truth={"files": ["src/b.cpp"]}
        ),
    ]
    compiled_calls: list[list[str]] = []
    planner_predictors: list[object | None] = []

    class FakeYAMSClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePlanner:
        def __init__(self, yams, **kwargs):
            planner_predictors.append(kwargs.get("dspy_rerank_predictor"))

        async def execute(self, specs, allow_adaptive=True):
            return []

    async def fake_run_task(task, **kwargs):
        return {
            "task_id": task.id,
            "task_type": task.task_type.value,
            "spec_count": 1,
            "metrics": {"file_hit_at_1": 0.0, "retrieval_latency_ms": 1.0},
        }

    monkeypatch.setattr(ret_bench, "YAMSClient", FakeYAMSClient)
    monkeypatch.setattr(ret_bench, "QueryPlanner", FakePlanner)
    monkeypatch.setattr(ret_bench, "_run_task", fake_run_task)
    monkeypatch.setattr(
        ret_bench,
        "_build_compiled_dspy_reranker",
        lambda config, train_tasks: (
            compiled_calls.append([t.id for t in train_tasks])
            or {"compiled_for": [t.id for t in train_tasks]}
        ),
    )

    cfg = PipelineConfig(
        yams_cwd="/tmp",
        use_dspy_retrieval_rerank=True,
        dspy_retrieval_optimize=True,
        dspy_retrieval_model=ModelConfig(name="openai/gpt-oss-20b"),
    )

    rows = await ret_bench._run_suite(
        tasks,
        cfg,
        max_queries=3,
        decompose_mode="heuristic",
        use_task_seeding=False,
        decomposer_temperature=0.0,
    )

    assert [row["task_id"] for row in rows] == ["a", "b"]
    assert compiled_calls == [["b"], ["a"]]
    assert planner_predictors[-2:] == [{"compiled_for": ["b"]}, {"compiled_for": ["a"]}]


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


def test_eval_runner_loads_plan_prompt_from_task_yaml(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(
        """
id: t1
task_type: coding
description: Explain the change
evaluation:
  plan: |
    Tests First
    1. Add a regression test
        """.strip(),
        encoding="utf-8",
    )
    runner = cov_bench.EvalRunner(PipelineConfig(), task_dir)
    tasks = runner.load_tasks(task_dir)
    assert len(tasks) == 1
    assert tasks[0].plan.startswith("Tests First")


def test_report_benchmark_summarizes_and_compares() -> None:
    rows = [
        {
            "model": "qwen-122b-executor",
            "task_id": "t1",
            "passed": True,
            "metrics": {"quality_score": 0.9, "total_latency_ms": 100.0},
        },
        {
            "model": "qwen35-35b-a3b",
            "task_id": "t1",
            "passed": False,
            "metrics": {"quality_score": 0.7, "total_latency_ms": 80.0},
        },
    ]
    summary = report_bench.summarize_models(rows)
    comparison = report_bench.compare_models(rows)
    assert summary["qwen-122b-executor"]["pass_rate_mean"] == 1.0
    assert comparison["wins"]["qwen-122b-executor"] == 1
