from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from dcs import cli
from dcs.types import EvalResult, EvalTask, ModelConfig, PipelineConfig, PipelineResult, TaskType


def test_read_yaml_and_search_weights(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    assert cli._read_yaml(missing) == {}

    cfg_path = tmp_path / "search_weights.yaml"
    cfg_path.write_text(
        """
search:
  fusion_weights:
    semantic: 0.6
    grep: bad
""".strip(),
        encoding="utf-8",
    )
    assert cli._load_search_weights(tmp_path) == {"semantic": 0.6}


def test_load_models_and_pipeline_config(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        """
backends:
  lm:
    base_url: http://localhost:1234/v1
    api_key: token
defaults:
  executor: ex
  critic: cr
models:
  ex:
    name: executor-model
    backend: lm
    context_window: 32000
  cr:
    name: critic-model
    backend: lm
    temperature: 0.2
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "search_weights.yaml").write_text(
        "search: {fusion_weights: {semantic: 0.7}}", encoding="utf-8"
    )

    executor, critic = cli._load_models(tmp_path)
    assert executor.name == "executor-model"
    assert executor.context_window == 32000
    assert critic is not None and critic.name == "critic-model"

    cfg = cli.load_pipeline_config(tmp_path)
    assert cfg.search_weights == {"semantic": 0.7}


def test_apply_runtime_overrides_and_resolve_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = argparse.Namespace(
        ground_truth_mode=False,
        dspy_faithfulness=False,
        context_profile="large",
    )
    cfg = cli._apply_runtime_overrides(args, PipelineConfig())
    assert cfg.no_ground_truth_mode is False
    assert cfg.use_dspy_faithfulness is False
    assert cfg.context_profile == "large"

    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    preferred.mkdir()
    fallback.mkdir()

    monkeypatch.setenv("DCS_TEST_PATH", str(fallback))
    assert cli._resolve_path("DCS_TEST_PATH", preferred, tmp_path) == fallback
    monkeypatch.delenv("DCS_TEST_PATH")
    assert cli._resolve_path("DCS_TEST_PATH", preferred, fallback) == preferred


def test_build_parser_parses_subcommands(tmp_path: Path) -> None:
    parser = cli._build_parser(tmp_path)
    run_args = parser.parse_args(["run", "hello"])
    eval_args = parser.parse_args(["eval", "--type", "qa"])
    review_args = parser.parse_args(["review", "--plan", "1. do x"])
    status_args = parser.parse_args(["status"])

    assert run_args.cmd == "run"
    assert eval_args.cmd == "eval"
    assert review_args.cmd == "review"
    assert status_args.cmd == "status"


@pytest.mark.asyncio
async def test_cmd_run_prints_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        async def run(self, task: str) -> PipelineResult:
            return PipelineResult(task=task, final_output="done")

    monkeypatch.setattr(cli, "DCSPipeline", FakePipeline)
    console = Console(record=True)
    rc = await cli._cmd_run(argparse.Namespace(task="x"), PipelineConfig(), console)
    assert rc == 0
    assert "done" in console.export_text()


@pytest.mark.asyncio
async def test_cmd_eval_and_compare_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeRunner:
        def __init__(self, cfg, task_dir):
            self.cfg = cfg
            self.task_dir = task_dir

        def load_tasks(self, task_dir, task_type=None):
            if str(task_dir).endswith("empty"):
                return []
            return [EvalTask(id="t", task_type=TaskType.QA, description="d")]

        async def run_suite(self, tasks, scaffolded=True):
            calls.append("run_suite")
            return [EvalResult(task_id="t", passed=True, metrics={})]

        def print_results(self, results, title):
            calls.append(f"print:{title}")

    async def fake_report(runner, tasks):
        calls.append("compare_report")
        return None

    fake_mod = types.SimpleNamespace(EvalRunner=FakeRunner, run_comparison_report=fake_report)
    monkeypatch.setitem(sys.modules, "eval.runner", fake_mod)

    console = Console(record=True)
    cfg = PipelineConfig()

    rc_no_tasks = await cli._cmd_eval(
        argparse.Namespace(task_dir=str(tmp_path / "empty"), type=None),
        cfg,
        console,
    )
    assert rc_no_tasks == 1

    rc_eval = await cli._cmd_eval(
        argparse.Namespace(task_dir=str(tmp_path / "full"), type="qa"),
        cfg,
        console,
    )
    assert rc_eval == 0

    rc_compare = await cli._cmd_compare(
        argparse.Namespace(task_dir=str(tmp_path / "full")), cfg, console
    )
    assert rc_compare == 0
    assert "run_suite" in calls
    assert "compare_report" in calls


@pytest.mark.asyncio
async def test_cmd_review_reads_inputs_and_writes_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("1. add review mode", encoding="utf-8")
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text("updated cli and reviewer", encoding="utf-8")
    out_file = tmp_path / "review.json"

    class FakeReviewer:
        def __init__(self, cfg):
            self.cfg = cfg

        async def review(self, review_input):
            assert review_input.plan == "1. add review mode"
            assert review_input.change_summary == "updated cli and reviewer"
            from dcs.types import PlanReviewResult

            return PlanReviewResult(
                task="demo",
                coverage_score=0.9,
                executed_well=True,
                summary="looks good",
                suggested_tests=["test review command"],
            )

    monkeypatch.setattr(cli, "PlanReviewer", FakeReviewer)
    console = Console(record=True)
    rc = await cli._cmd_review(
        argparse.Namespace(
            task="demo",
            plan="",
            plan_file=str(plan_file),
            diff_file=None,
            change_summary="",
            change_summary_file=str(summary_file),
            execution_summary="",
            execution_summary_file=None,
            changed_files="a.py,b.py",
            json_out=str(out_file),
        ),
        PipelineConfig(),
        console,
    )
    assert rc == 0
    assert out_file.exists()
    assert "looks good" in console.export_text()


@pytest.mark.asyncio
async def test_cmd_status_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def status(self):
            return {"ok": True}

    class GoodExecutor:
        def __init__(self, cfg):
            self.cfg = cfg

        async def health_check(self):
            return True

    class BadExecutor:
        def __init__(self, cfg):
            self.cfg = cfg

        async def health_check(self):
            return False

    monkeypatch.setattr(cli, "_init_yams_client", lambda cfg: FakeClient())
    monkeypatch.setattr(cli, "ModelExecutor", GoodExecutor)

    console = Console(record=True)
    ok = await cli._cmd_status(argparse.Namespace(), PipelineConfig(), console)
    assert ok == 0

    monkeypatch.setattr(cli, "ModelExecutor", BadExecutor)
    bad = await cli._cmd_status(argparse.Namespace(), PipelineConfig(), console)
    assert bad == 1


def test_main_dispatches_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_pipeline_config", lambda _: PipelineConfig())
    monkeypatch.setattr(cli, "_resolve_path", lambda env, preferred, fallback: preferred)

    async def fake_run(args, cfg, console):
        return 7

    monkeypatch.setattr(cli, "_cmd_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["research-agent", "run", "hello"])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 7


def test_init_yams_client_typeerror_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYAMSClient:
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("unsupported kwargs")
            self.yams_binary = ""
            self.yams_data_dir = ""

    fake_mod = types.SimpleNamespace(YAMSClient=FakeYAMSClient)
    monkeypatch.setitem(sys.modules, "dcs.client", fake_mod)

    cfg = PipelineConfig(
        executor_model=ModelConfig(name="m"), yams_binary="yams", yams_data_dir="data"
    )
    client = cli._init_yams_client(cfg)
    assert getattr(client, "yams_binary", "") == "yams"
    assert getattr(client, "yams_data_dir", "") == "data"
