from __future__ import annotations

import re

from dcs.types import ContextBlock, Critique, EvalTask, PipelineResult


def _token_count(text: str) -> int:
    # Cheap tokenizer; good enough for eval thresholds.
    return len((text or "").strip().split())


def score_contains_pattern(output: str, patterns: list[str]) -> float:
    """Fraction of patterns found in output.

    Patterns support:
    - literal substring (case-insensitive)
    - regex via prefix 're:'
    """

    pats = [p for p in (patterns or []) if isinstance(p, str) and p.strip()]
    if not pats:
        return 0.0

    out = output or ""
    found = 0
    for p in pats:
        p = p.strip()
        if p.lower().startswith("re:"):
            rx = p[3:].strip()
            try:
                if re.search(rx, out, flags=re.IGNORECASE | re.MULTILINE):
                    found += 1
            except re.error:
                # Treat invalid regex as not found.
                continue
        else:
            if p.lower() in out.lower():
                found += 1
    return found / len(pats)


def score_exact_match(output: str, expected: str) -> float:
    if (output or "").strip() == (expected or "").strip():
        return 1.0
    return 0.0


def score_output_length(output: str, min_tokens: int, max_tokens: int) -> float:
    toks = _token_count(output)
    if toks < int(min_tokens):
        return 0.0
    if toks > int(max_tokens):
        return 0.0
    return 1.0


def context_efficiency(context: ContextBlock, critique: Critique) -> float:
    """Proxy for useful tokens / total.

    We combine the critic's stated utilization signal with chunk-level relevance.
    """

    if context is None or critique is None:
        return 0.0
    total = int(context.chunks_included or len(context.chunk_ids or []))
    irrelevant = len(critique.irrelevant_chunks or [])
    if total <= 0:
        return float(critique.context_utilization or 0.0)
    relevant_fraction = (total - irrelevant) / total
    return max(0.0, min(1.0, relevant_fraction * float(critique.context_utilization or 0.0)))


def retrieval_precision(critique: Critique, context: ContextBlock) -> float:
    if context is None or critique is None:
        return 0.0
    total = int(context.chunks_included or len(context.chunk_ids or []))
    if total <= 0:
        return 0.0
    irrelevant = len(critique.irrelevant_chunks or [])
    return max(0.0, min(1.0, (total - irrelevant) / total))


def evaluate_task(task: EvalTask, result: PipelineResult) -> dict[str, float]:
    """Evaluate a PipelineResult using task-provided evaluation hints.

    Expected YAML keys (flexible):
    - ground_truth.expected / evaluation.expected
    - evaluation.contains_patterns / evaluation.patterns
    - evaluation.length: {min_tokens, max_tokens}
    """

    metrics: dict[str, float] = {}
    out = (result.final_output or "").strip()

    final_it = result.iterations[-1] if result.iterations else None
    best_it = None
    best_num = int(getattr(result, "best_iteration", 0) or 0)
    if best_num > 0:
        best_it = next((it for it in result.iterations if int(it.iteration) == best_num), None)
    it = best_it or final_it
    context = it.context if it else None
    critique = it.critique if it else None

    gt = task.ground_truth or {}
    ev = task.evaluation or {}

    expected = ev.get("expected") or gt.get("expected") or gt.get("exact")
    if isinstance(expected, str):
        metrics["exact_match"] = score_exact_match(out, expected)

    patterns = ev.get("contains_patterns") or ev.get("patterns") or gt.get("patterns")
    if isinstance(patterns, list):
        metrics["contains_pattern"] = score_contains_pattern(out, [str(p) for p in patterns])

    length_cfg = ev.get("length") or {}
    if isinstance(length_cfg, dict) and ("min_tokens" in length_cfg or "max_tokens" in length_cfg):
        min_t = int(length_cfg.get("min_tokens") or 0)
        max_t = int(length_cfg.get("max_tokens") or 10**9)
        metrics["output_length"] = score_output_length(out, min_t, max_t)

    if critique is not None:
        metrics["quality_score"] = float(critique.quality_score or 0.0)

    if context is not None and critique is not None:
        metrics["context_efficiency"] = context_efficiency(context, critique)
        metrics["retrieval_precision"] = retrieval_precision(critique, context)
    else:
        metrics["context_efficiency"] = 0.0
        metrics["retrieval_precision"] = 0.0

    # Ground-truth-free faithfulness signals (if enabled in pipeline).
    faith = getattr(it, "faithfulness", None) if it else None
    if faith is not None:
        metrics["faithfulness_confidence"] = float(getattr(faith, "confidence", 0.0) or 0.0)
        metrics["faithfulness_supported_ratio"] = float(
            getattr(faith, "supported_ratio", 0.0) or 0.0
        )
        metrics["faithfulness_should_abstain"] = (
            1.0 if bool(getattr(faith, "should_abstain", False)) else 0.0
        )
    else:
        metrics["faithfulness_confidence"] = 0.0
        metrics["faithfulness_supported_ratio"] = 0.0
        metrics["faithfulness_should_abstain"] = 0.0

    metrics["iterations"] = float(len(result.iterations or []))
    metrics["total_latency_ms"] = float(result.total_latency_ms or 0.0)

    plan_review = getattr(result, "plan_review", None)
    if plan_review is not None:
        metrics["plan_coverage"] = float(getattr(plan_review, "coverage_score", 0.0) or 0.0)
        metrics["plan_executed_well"] = (
            1.0 if bool(getattr(plan_review, "executed_well", False)) else 0.0
        )
    return metrics
