# DCS Execution Plan (Accuracy-First)

## Goal
Improve grounded accuracy and retrieval reliability while controlling paid-model spend.

## Success Metrics
- Grounded pass rate (QA/coding task suites)
- Escalation rate (fraction of tasks requiring stronger/paid tier)
- Cost per successful task (prompt + completion tokens)
- Retrieval coverage depth (distinct authoritative sources used)

## Phase 1: Overflow-Aware Execution
1. Parse context-overflow 400 errors in executor.
2. Retry with bounded shrink loop in pipeline.
3. Shrink both context budget and output tokens.
4. Keep non-droppable prompt parts intact (system/task/codemap header).

Guardrails:
- Max retries: `max_context_overflow_retries`
- Min context: `min_context_budget`
- Min output tokens: `min_output_tokens`

## Phase 2: Query-Type Adaptive Decomposition
1. Use optimizer query-type scores per iteration.
2. Bias decomposer toward GREP/GET when semantic underperforms.
3. Enforce minimum required query types for robustness.
4. Keep at least one semantic probe to avoid mode collapse.

## Phase 3: Tiered Escalation Router
1. Run local scaffold first.
2. Escalate to stronger model only when quality/coverage gates fail.
3. Prefer answer-only escalation before full re-retrieval.
4. Enforce escalation caps per task.

## Phase 4: YAMS MCP UX Improvements
Quick wins:
- Search anchors + truncation markers + score breakdown.
- Structured grep `matches[]` alongside legacy text output.
- `get` truncation + metadata fields.

## Phase 5: Evaluation
1. Run checkpointed QA sweep across model tiers.
2. Compare against baseline:
   - grounded pass rate
   - escalation rate
   - cost/success
   - retrieval coverage depth
3. Tune thresholds and retry/bias factors based on results.

## Phase 6: Ingest-First Retrieval + Plan Review
1. Prime YAMS indexing before retrieval and benchmark runs.
2. Validate search anchors before expanding graph traversal.
3. Use graph only to refine trusted anchors into better GET/GREP targets.
4. Add post-execution plan review as a separate evaluation axis.

Acceptance:
- Retrieval benchmarks run on current ingested repo state.
- Graph expansion improves authoritative file targeting instead of inflating context.
- Plan review produces actionable gaps, tests, and follow-up advice.
