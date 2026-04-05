from __future__ import annotations

from dcs.plan_review import (
    _counts_for_coverage,
    _looks_like_rich_plan_prompt,
    PlanReviewer,
    build_change_summary,
    parse_plan_steps,
)
from dcs.types import ContextBlock, PipelineConfig, PlanReviewInput, PlanStep, PlanStepStatus


def test_parse_plan_steps_handles_bullets() -> None:
    steps = parse_plan_steps("1. add review mode\n2. benchmark it")
    assert len(steps) == 2
    assert steps[0].description == "add review mode"
    assert steps[1].step_id == "step-2"


def test_parse_plan_steps_handles_paragraphs() -> None:
    steps = parse_plan_steps("Add review mode\n\nAdd benchmark report")
    assert len(steps) == 2


def test_parse_plan_steps_handles_structured_sections_and_reminders() -> None:
    steps = parse_plan_steps(
        """
Assumption
- Keep CompressedANN separate for now

Tests First
1. Add a fusion-contract unit test
- assert source string
- assert weight mapping

Acceptance Gates
- No regression in TurboQuant-only behavior

<system-reminder>
Your operational mode has changed from plan to build.
</system-reminder>
        """
    )
    assert [step.section for step in steps] == ["Assumption", "Tests First", "Acceptance Gates"]
    assert [step.step_type for step in steps] == ["assumption", "required_step", "acceptance_gate"]
    assert steps[0].description == "Keep CompressedANN separate for now"
    assert steps[1].acceptance_criteria == ["assert source string", "assert weight mapping"]
    assert "system-reminder" not in "\n".join(step.description for step in steps)


def test_build_change_summary_ignores_system_reminder_in_plan_input() -> None:
    summary = build_change_summary(
        PlanReviewInput(
            plan="task body\n<system-reminder>build mode</system-reminder>",
            task="compare models",
            change_summary="updated benchmark command",
        )
    )
    assert "updated benchmark command" in summary


def test_rich_plan_prompt_detection_and_auto_plan_fallback() -> None:
    prompt = """
Tests First
1. Add a regression test
- assert exact behavior

Acceptance Gates
- no regressions

<system-reminder>
Your operational mode has changed from plan to build.
</system-reminder>
    """
    assert _looks_like_rich_plan_prompt(prompt) is True

    reviewer = PlanReviewer(PipelineConfig())
    normalized = reviewer._normalize_input(PlanReviewInput(plan="", task=prompt))
    assert normalized.plan
    assert normalized.task == "Add a regression test"
    assert "system-reminder" not in normalized.plan


def test_parse_normalized_plan_response_preserves_types() -> None:
    reviewer = PlanReviewer(PipelineConfig())
    task, steps = reviewer._parse_normalized_plan_response(
        {
            "task_summary": "Validate compressed ANN changes",
            "steps": [
                {
                    "section": "Acceptance Gates",
                    "step_type": "acceptance_gate",
                    "description": "No regression in TurboQuant-only behavior",
                    "acceptance_criteria": ["MRR delta >= -0.002"],
                }
            ],
        }
    )
    assert task == "Validate compressed ANN changes"
    assert len(steps) == 1
    assert steps[0].section == "Acceptance Gates"
    assert steps[0].step_type == "acceptance_gate"
    assert steps[0].acceptance_criteria == ["MRR delta >= -0.002"]


def test_parse_plan_steps_handles_status_and_blocker_sections() -> None:
    steps = parse_plan_steps(
        """
What's covered now:
- compressed_ann_fusion_contract_catch2_test.cpp proves the source contract
- compressed_ann_fusion_behavior_catch2_test.cpp proves fusion behavior

What still blocks end-to-end confidence:
- pre-fusion dedup at src/search/search_engine.cpp:219

Benchmark Cases To Rerun
1. SciFact 25-query A/B/C
        """
    )
    assert [step.section for step in steps] == [
        "What's covered now",
        "What's covered now",
        "What still blocks end-to-end confidence",
        "Benchmark Cases To Rerun",
    ]
    assert [step.step_type for step in steps] == ["evidence", "evidence", "gap", "benchmark"]


def test_derive_task_ignores_evidence_sections() -> None:
    prompt = """
What's covered now:
- contract test is green

What still blocks end-to-end confidence:
- dedup path is unverified

Tests First
1. Add the pre-fusion dedup regression test
    """
    reviewer = PlanReviewer(PipelineConfig())
    normalized = reviewer._normalize_input(PlanReviewInput(plan="", task=prompt))
    assert normalized.task == "Add the pre-fusion dedup regression test"


def test_coverage_helper_ignores_evidence_and_gap_steps() -> None:
    assert (
        _counts_for_coverage(PlanStep(step_id="1", description="done", step_type="evidence"))
        is False
    )
    assert (
        _counts_for_coverage(PlanStep(step_id="2", description="blocked", step_type="gap")) is False
    )
    assert (
        _counts_for_coverage(
            PlanStep(step_id="3", description="rerun benchmark", step_type="benchmark")
        )
        is True
    )


def test_heuristic_review_uses_benchmark_reruns_not_regression_tests() -> None:
    reviewer = PlanReviewer(PipelineConfig())
    result = reviewer._heuristic_review(
        PlanReviewInput(plan="", task="t", execution_summary="reran scifact and fixed dedup"),
        [
            PlanStep(step_id="step-1", description="SciFact 25-query A/B/C", step_type="benchmark"),
            PlanStep(step_id="step-2", description="What's covered now", step_type="evidence"),
            PlanStep(step_id="step-3", description="dedup path still blocked", step_type="gap"),
        ],
        [],
        ContextBlock(content="reran scifact and fixed dedup"),
    )
    assert result.suggested_tests == ["Rerun benchmark case: SciFact 25-query A/B/C"]
    assert result.followup_plan[0] == "dedup path still blocked"


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
