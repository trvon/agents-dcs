from __future__ import annotations

from dcs.plan_review import build_change_summary, parse_plan_steps
from dcs.types import PlanReviewInput, PlanStepStatus


def test_parse_plan_steps_handles_bullets() -> None:
    steps = parse_plan_steps("1. add review mode\n2. benchmark it")
    assert len(steps) == 2
    assert steps[0].description == "add review mode"
    assert steps[1].step_id == "step-2"


def test_parse_plan_steps_handles_paragraphs() -> None:
    steps = parse_plan_steps("Add review mode\n\nAdd benchmark report")
    assert len(steps) == 2


def test_build_change_summary_includes_files_and_sections() -> None:
    summary = build_change_summary(
        PlanReviewInput(
            plan="1. compare models",
            task="compare models",
            change_summary="updated benchmark command",
            execution_summary="executor finished",
            changed_files=["dcs/cli.py", "benchmarks/report_benchmark.py"],
        )
    )
    assert "compare models" in summary
    assert "dcs/cli.py" in summary
    assert "updated benchmark command" in summary


def test_plan_step_status_values_are_stable() -> None:
    assert PlanStepStatus.COMPLETE.value == "complete"
    assert PlanStepStatus.MISSING.value == "missing"
