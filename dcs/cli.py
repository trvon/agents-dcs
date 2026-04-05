from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dcs.executor import ModelExecutor
from dcs.pipeline import DCSPipeline
from dcs.plan_review import PlanReviewer, _looks_like_rich_plan_prompt
from dcs.runtime_config import load_runtime_settings
from dcs.types import ModelConfig, PipelineConfig, TaskType


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_models(config_dir: Path) -> tuple[ModelConfig, ModelConfig | None]:
    cfg = _read_yaml(config_dir / "models.yaml")
    backends = cfg.get("backends") or {}
    models = cfg.get("models") or {}
    defaults = cfg.get("defaults") or {}

    def build(key: str) -> ModelConfig:
        entry = models.get(key) or {}
        backend_key = entry.get("backend")
        backend = backends.get(backend_key) if isinstance(backends, dict) else None
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        if isinstance(backend, dict):
            base_url = str(backend.get("base_url") or base_url)
            api_key = str(backend.get("api_key") or api_key)
        return ModelConfig(
            name=str(entry.get("name") or key),
            base_url=base_url,
            api_key=api_key,
            context_window=int(entry.get("context_window") or 4096),
            max_output_tokens=int(entry.get("max_output_tokens") or 1024),
            temperature=float(entry.get("temperature") or 0.7),
            system_suffix=str(entry.get("system_suffix") or ""),
            request_timeout_s=float(entry.get("request_timeout_s") or 600.0),
            max_retries=int(entry.get("max_retries") or 2),
            retry_backoff_s=float(entry.get("retry_backoff_s") or 2.0),
        )

    executor_key = str(defaults.get("executor") or "")
    critic_key = str(defaults.get("critic") or "")

    executor = (
        build(executor_key) if executor_key else ModelConfig(name="qwen/qwen3-4b-thinking-2507")
    )
    critic = build(critic_key) if critic_key else None
    return executor, critic


def _load_search_weights(config_dir: Path) -> dict[str, float]:
    cfg = _read_yaml(config_dir / "search_weights.yaml")
    search = cfg.get("search") or {}
    fusion = {}
    if isinstance(search, dict):
        fusion = search.get("fusion_weights") or search.get("fusion") or {}
    out: dict[str, float] = {}
    if isinstance(fusion, dict):
        for k, v in fusion.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    return out


def load_pipeline_config(config_dir: Path) -> PipelineConfig:
    executor, critic = _load_models(config_dir)
    weights = _load_search_weights(config_dir)
    return PipelineConfig(executor_model=executor, critic_model=critic, search_weights=weights)


def _apply_runtime_overrides(args: argparse.Namespace, cfg: PipelineConfig) -> PipelineConfig:
    ngt = getattr(args, "ground_truth_mode", None)
    if ngt is not None:
        cfg.no_ground_truth_mode = bool(ngt)
    dspy_flag = getattr(args, "dspy_faithfulness", None)
    if dspy_flag is not None:
        cfg.use_dspy_faithfulness = bool(dspy_flag)
    profile = getattr(args, "context_profile", None)
    if profile:
        cfg.context_profile = str(profile)
    return cfg


def _read_optional_text(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def _split_changed_files(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in str(raw).split(","):
        s = item.strip()
        if s and s not in out:
            out.append(s)
    return out


def _init_yams_client(cfg: PipelineConfig):
    from dcs.client import YAMSClient

    try:
        return YAMSClient(yams_binary=cfg.yams_binary, yams_data_dir=cfg.yams_data_dir)  # type: ignore[arg-type]
    except TypeError:
        c = YAMSClient()  # type: ignore[call-arg]
        if hasattr(c, "yams_binary"):
            try:
                c.yams_binary = cfg.yams_binary
            except Exception:
                pass
        if hasattr(c, "yams_data_dir"):
            try:
                c.yams_data_dir = cfg.yams_data_dir
            except Exception:
                pass
        return c


async def _cmd_run(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    pipe = DCSPipeline(cfg)
    res = await pipe.run(args.task)
    console.rule("Output")
    console.print(res.final_output)
    return 0


async def _cmd_eval(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    try:
        from eval.runner import EvalRunner
    except Exception as e:
        console.print(f"Eval runner import failed: {e}")
        return 2

    task_dir = Path(args.task_dir)
    runner = EvalRunner(cfg, task_dir)
    ttype = TaskType(args.type) if args.type else None
    tasks = runner.load_tasks(task_dir, task_type=ttype)
    if not tasks:
        console.print(f"No tasks found in {task_dir}")
        return 1

    results = await runner.run_suite(tasks, scaffolded=True)
    runner.print_results(results, "Evaluation")
    return 0


async def _cmd_compare(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    try:
        from eval.runner import EvalRunner, run_comparison_report
    except Exception as e:
        console.print(f"Eval runner import failed: {e}")
        return 2

    task_dir = Path(args.task_dir)
    runner = EvalRunner(cfg, task_dir)
    tasks = runner.load_tasks(task_dir)
    if not tasks:
        console.print(f"No tasks found in {task_dir}")
        return 1
    await run_comparison_report(runner, tasks)
    return 0


async def _cmd_review(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    plan = _read_optional_text(getattr(args, "plan_file", None))
    if not plan:
        plan = str(getattr(args, "plan", "") or "").strip()
    task_text = str(getattr(args, "task", "") or "").strip()
    if not plan and _looks_like_rich_plan_prompt(task_text):
        plan = task_text
    if not plan:
        console.print("Plan review requires --plan or --plan-file")
        return 2

    from dcs.types import PlanReviewInput

    review_input = PlanReviewInput(
        plan=plan,
        task=task_text,
        diff_text=_read_optional_text(getattr(args, "diff_file", None)),
        change_summary=_read_optional_text(getattr(args, "change_summary_file", None))
        or str(getattr(args, "change_summary", "") or "").strip(),
        execution_summary=_read_optional_text(getattr(args, "execution_summary_file", None))
        or str(getattr(args, "execution_summary", "") or "").strip(),
        changed_files=_split_changed_files(getattr(args, "changed_files", None)),
    )

    reviewer = PlanReviewer(cfg)
    result = await reviewer.review(review_input)

    console.rule("Plan Review")
    console.print(f"Coverage: {result.coverage_score:.2f}")
    console.print(f"Executed well: {'yes' if result.executed_well else 'no'}")
    if result.summary:
        console.print(result.summary)
    if result.gaps:
        console.print("Gaps:")
        for gap in result.gaps:
            console.print(f"- {gap}")
    if result.suggested_tests:
        console.print("Suggested tests:")
        for test in result.suggested_tests:
            console.print(f"- {test}")

    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        console.print(f"Wrote plan review to {args.json_out}")
    return 0


async def _cmd_status(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    tbl = Table(title="Status")
    tbl.add_column("Service")
    tbl.add_column("OK")
    tbl.add_column("Details")

    # YAMS
    yams_ok = False
    yams_detail = ""
    try:
        async with _init_yams_client(cfg) as c:
            st = await c.status()
            yams_ok = True
            yams_detail = str(st)
    except Exception as e:
        yams_detail = str(e)
    tbl.add_row("yams", "yes" if yams_ok else "no", yams_detail)

    # LM Studio/OpenAI-compatible
    llm_ok = False
    llm_detail = ""
    try:
        ex = ModelExecutor(cfg.executor_model)
        llm_ok = await ex.health_check()
        llm_detail = f"model={cfg.executor_model.name} base_url={cfg.executor_model.base_url}"
    except Exception as e:
        llm_detail = str(e)
    tbl.add_row("llm", "yes" if llm_ok else "no", llm_detail)

    console.print(tbl)
    return 0 if (yams_ok and llm_ok) else 1


def _build_parser(default_task_dir: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research-agent", description="Research Scaffold Agent (DCS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="run a single task")
    runp.add_argument("task", type=str, help="task description")
    runp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    runp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    runp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    evalp = sub.add_parser("eval", help="run evaluation suite")
    evalp.add_argument("--task-dir", type=str, default=str(default_task_dir))
    evalp.add_argument("--type", type=str, choices=[t.value for t in TaskType], default=None)
    evalp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    evalp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    evalp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    compp = sub.add_parser("compare", help="compare scaffolded vs vanilla")
    compp.add_argument("--task-dir", type=str, default=str(default_task_dir))
    compp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    compp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    compp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    reviewp = sub.add_parser("review", help="review whether changes satisfied a plan")
    reviewp.add_argument("--task", type=str, default="", help="task description")
    reviewp.add_argument("--plan", type=str, default="", help="inline plan text")
    reviewp.add_argument("--plan-file", type=str, default=None, help="path to plan text file")
    reviewp.add_argument("--diff-file", type=str, default=None, help="path to unified diff file")
    reviewp.add_argument(
        "--change-summary",
        type=str,
        default="",
        help="inline summary of code changes",
    )
    reviewp.add_argument(
        "--change-summary-file",
        type=str,
        default=None,
        help="path to change summary file",
    )
    reviewp.add_argument(
        "--execution-summary",
        type=str,
        default="",
        help="inline executor summary/output",
    )
    reviewp.add_argument(
        "--execution-summary-file",
        type=str,
        default=None,
        help="path to execution summary file",
    )
    reviewp.add_argument(
        "--changed-files",
        type=str,
        default="",
        help="comma-separated changed files",
    )
    reviewp.add_argument("--json-out", type=str, default=None, help="write JSON result to path")

    sub.add_parser("status", help="check YAMS + model connectivity")
    return p


def _resolve_path(env_var: str, preferred: Path, fallback: Path) -> Path:
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return Path(env_val).expanduser()
    if preferred.exists():
        return preferred
    return fallback


def main() -> None:
    console = Console()
    base_dir = Path(__file__).resolve().parents[1]
    runtime = load_runtime_settings(base_dir)
    config_dir = _resolve_path(
        "DCS_CONFIG_DIR",
        runtime.config_dir or (base_dir / "configs"),
        Path.cwd() / "configs",
    )
    default_task_dir = _resolve_path(
        "DCS_TASK_DIR",
        runtime.task_dir or (base_dir / "eval" / "tasks"),
        Path.cwd() / "eval" / "tasks",
    )

    parser = _build_parser(default_task_dir)
    args = parser.parse_args()

    cfg = _apply_runtime_overrides(args, load_pipeline_config(config_dir))
    if not cfg.yams_cwd and runtime.yams_cwd is not None:
        cfg.yams_cwd = str(runtime.yams_cwd)

    async def run_cmd() -> int:
        if args.cmd == "run":
            return await _cmd_run(args, cfg, console)
        if args.cmd == "eval":
            return await _cmd_eval(args, cfg, console)
        if args.cmd == "compare":
            return await _cmd_compare(args, cfg, console)
        if args.cmd == "review":
            return await _cmd_review(args, cfg, console)
        if args.cmd == "status":
            return await _cmd_status(args, cfg, console)
        console.print(f"Unknown command: {args.cmd}")
        return 2

    raise SystemExit(asyncio.run(run_cmd()))
