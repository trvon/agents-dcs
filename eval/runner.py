from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dcs.pipeline import DCSPipeline
from dcs.types import (
    ComparisonResult,
    EvalResult,
    EvalTask,
    PipelineConfig,
    PipelineResult,
    TaskType,
)
from eval.metrics import evaluate_task


def _as_task_type(v: Any) -> TaskType | None:
    if isinstance(v, TaskType):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        for t in TaskType:
            if t.value == s:
                return t
    return None


class EvalRunner:
    def __init__(self, config: PipelineConfig, task_dir: str | Path):
        self.config = config
        self.task_dir = Path(task_dir)
        self.console = Console()

    def load_tasks(self, task_dir: str | Path, task_type: TaskType | None = None) -> list[EvalTask]:
        p = Path(task_dir)
        if not p.exists() or not p.is_dir():
            return []

        tasks: list[EvalTask] = []
        for fp in sorted(list(p.rglob("*.yaml")) + list(p.rglob("*.yml"))):
            try:
                raw = yaml.safe_load(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue

            tid = raw.get("id") or fp.stem
            ttype = _as_task_type(raw.get("task_type") or raw.get("type") or raw.get("kind"))
            if ttype is None:
                continue
            if task_type is not None and ttype != task_type:
                continue

            desc = raw.get("description") or raw.get("task") or ""
            if not isinstance(desc, str):
                desc = str(desc)

            ground_truth = raw.get("ground_truth") or {}
            evaluation = raw.get("evaluation") or {}
            tags = raw.get("tags") or []

            plan = raw.get("plan") or ""
            if not plan and isinstance(evaluation, dict):
                plan = evaluation.get("plan") or evaluation.get("plan_prompt") or ""
            if not isinstance(plan, str):
                plan = str(plan)

            if not isinstance(ground_truth, dict):
                ground_truth = {}
            if not isinstance(evaluation, dict):
                evaluation = {}
            if not isinstance(tags, list):
                tags = [str(tags)]

            tasks.append(
                EvalTask(
                    id=str(tid),
                    task_type=ttype,
                    description=desc.strip(),
                    plan=plan.strip(),
                    ground_truth=ground_truth,
                    evaluation=evaluation,
                    tags=[str(t).strip() for t in tags if str(t).strip()],
                )
            )
        return tasks

    def _decide_pass(self, task: EvalTask, metrics: dict[str, float]) -> bool:
        ev = task.evaluation or {}
        pass_metric = ev.get("pass_metric")
        threshold = float(ev.get("pass_threshold") or 1.0)
        if isinstance(pass_metric, str) and pass_metric in metrics:
            return float(metrics.get(pass_metric) or 0.0) >= threshold

        # Defaults: prefer exact-match, then contains-pattern, else quality.
        if "exact_match" in metrics:
            return metrics["exact_match"] >= 1.0
        if "contains_pattern" in metrics:
            return metrics["contains_pattern"] >= 1.0
        if "faithfulness_confidence" in metrics:
            fthr = float(ev.get("faithfulness_threshold") or 0.6)
            abstain = float(metrics.get("faithfulness_should_abstain") or 0.0)
            return float(metrics.get("faithfulness_confidence") or 0.0) >= fthr and abstain < 0.5
        if "quality_score" in metrics:
            return metrics["quality_score"] >= float(ev.get("quality_threshold") or 0.7)
        return False

    async def run_task(self, task: EvalTask, scaffolded: bool = True) -> EvalResult:
        pipe = DCSPipeline(self.config)
        try:
            if scaffolded:
                pr: PipelineResult = await pipe.run(task.description)
            else:
                pr = await pipe.run_vanilla(task.description)
            metrics = evaluate_task(task, pr)
            passed = self._decide_pass(task, metrics)
            return EvalResult(
                task_id=task.id,
                pipeline_result=pr,
                metrics=metrics,
                passed=passed,
                task_type=task.task_type.value,
                tags=list(task.tags or []),
                repeat_index=1,
            )
        except Exception as e:
            return EvalResult(
                task_id=task.id,
                pipeline_result=None,
                metrics={},
                passed=False,
                error=str(e),
                task_type=task.task_type.value,
                tags=list(task.tags or []),
                repeat_index=1,
            )

    async def run_suite(self, tasks: list[EvalTask], scaffolded: bool = True) -> list[EvalResult]:
        results: list[EvalResult] = []
        for t in tasks:
            r = await self.run_task(t, scaffolded=scaffolded)
            results.append(r)
        return results

    async def run_comparison(
        self, tasks: list[EvalTask]
    ) -> tuple[ComparisonResult, ComparisonResult]:
        exec_model = self.config.executor_model.name
        a = ComparisonResult(config_name="default", model=exec_model, scaffolded=True)
        b = ComparisonResult(config_name="default", model=exec_model, scaffolded=False)

        a.tasks = await self.run_suite(tasks, scaffolded=True)
        b.tasks = await self.run_suite(tasks, scaffolded=False)
        return a, b

    def print_results(self, results: list[EvalResult], title: str) -> None:
        tbl = Table(title=title)
        tbl.add_column("Task")
        tbl.add_column("Pass", justify="center")
        tbl.add_column("Quality", justify="right")
        tbl.add_column("Latency ms", justify="right")
        tbl.add_column("Iters", justify="right")

        for r in results:
            q = r.metrics.get("quality_score", 0.0)
            lat = r.metrics.get("total_latency_ms", 0.0)
            iters = r.metrics.get("iterations", 0.0)
            tbl.add_row(
                r.task_id, "yes" if r.passed else "no", f"{q:.2f}", f"{lat:.0f}", f"{iters:.0f}"
            )
        self.console.print(tbl)


async def run_comparison_report(
    runner: EvalRunner, tasks: list[EvalTask]
) -> tuple[ComparisonResult, ComparisonResult]:
    a, b = await runner.run_comparison(tasks)
    runner.print_results(a.tasks, "Scaffolded")
    runner.print_results(b.tasks, "Vanilla")

    sum_tbl = Table(title="Comparison Summary")
    sum_tbl.add_column("Variant")
    sum_tbl.add_column("Success", justify="right")
    sum_tbl.add_row("scaffolded", f"{a.success_rate:.2f}")
    sum_tbl.add_row("vanilla", f"{b.success_rate:.2f}")
    runner.console.print(sum_tbl)
    return a, b
