# Agent Architecture V2 (Reliability + Usefulness)

## Why this revision

Recent RLM/retrieval test runs show session instability (tool process exits, daemon readiness races, long-run retries). The agent is strong on retrieval logic, but weak on runtime durability.

This architecture keeps accuracy-first retrieval, while adding supervision, checkpoints, and deterministic recovery.

## Design goals

1. Reliability first: a failed subprocess should not kill the full task session.
2. Grounded outputs: answers must cite retrieved evidence.
3. Predictable behavior: staged retrieval and explicit gates.
4. Operator visibility: traces, health, and per-stage metrics.
5. Fast recovery: resume from last successful stage after crash.

## Layered architecture

1. Runtime Supervisor
   - Owns process health for YAMS MCP + model endpoint connectivity.
   - Maintains state machine: `starting -> ready -> degraded -> recovering -> failed`.
   - Applies circuit breakers and restart budget.

2. Session Orchestrator
   - Runs task lifecycle as explicit stages:
     - `decompose -> retrieve -> assemble -> execute -> critique -> finalize`
   - Persists stage checkpoints and run metadata.
   - Can replay from last successful stage.

3. Retrieval Engine
   - Uses staged order:
     - `prime index -> search -> validate anchors -> graph refine -> targeted grep + get`
   - Adds coverage controller:
     - if file coverage is low, force path-first retries.
   - Emits step traces (args, counts, filtered reasons, selected sources).

4. Evidence and Answer Layer
   - Builds answer only from cited evidence chunks.
   - Runs faithfulness gate before accepting final output.
   - Uses abstain mode when support is insufficient.

5. Evaluation and Policy Layer
   - Benchmarks retrieval and final task outcomes.
   - Keeps plan-review evaluation separate from answer-generation evaluation.
   - Applies strict acceptance gates (for this track: per-task recall@10 target).
   - Tracks regressions by task family.

## Component mapping to current code

- Keep and harden:
  - `dcs/decomposer.py`
  - `dcs/planner.py`
  - `dcs/assembler.py`
  - `dcs/executor.py`
  - `dcs/critic.py`
  - `dcs/faithfulness.py`

- Introduce or expand:
  - `dcs/runtime_supervisor.py` (new): health checks, restart policy, circuit breaker.
  - `dcs/session_store.py` (new): run checkpoints + replay metadata.
  - `dcs/retrieval_trace.py` (new): normalized trace events and artifact writer.
  - `dcs/pipeline.py` (expand): explicit checkpoint boundaries and resume path.

## Failure-handling contract

For each stage:

1. Preflight check
   - Verify dependency readiness before stage execution.
2. Execute with bounded retry
   - Retry only idempotent operations.
3. Persist checkpoint
   - Write stage result + diagnostics.
4. On failure
   - Classify (`transient`, `dependency`, `logic`, `budget`).
   - Attempt auto-recovery for `transient`/`dependency` within budget.
   - Otherwise fail fast with actionable reason.

## Minimal persistence schema

- `run_id`
- `task_id`
- `stage`
- `status`
- `attempt`
- `inputs_hash`
- `outputs_ref`
- `error_class`
- `error_message`
- `timestamp`

This is enough for resume, replay, and postmortem.

## Build order (implementation slices)

### Slice A: Runtime stability

1. Add `RuntimeSupervisor` with health probes for YAMS and model endpoint.
2. Switch pipeline startup to supervisor-managed readiness gates.
3. Add restart/circuit-breaker policy and surfaced diagnostics.

Acceptance:
- Session survives transient YAMS/model disconnects.
- Errors report dependency state clearly.

### Slice B: Checkpoint + resume

1. Add `SessionStore` (JSONL or sqlite) for stage checkpoints.
2. Persist checkpoints after each stage in `DCSPipeline`.
3. Add `--resume <run_id>` in CLI.

Acceptance:
- Interrupted run resumes from last completed stage.

### Slice C: Retrieval trace + coverage controller

1. Emit per-step retrieval trace artifacts.
2. Add coverage-driven retries before finalizing retrieval set.
3. Export per-task recall diagnostics.

Acceptance:
- Every retrieval miss is explainable from artifacts.

### Slice D: Production usefulness

1. Add task-intent templates (QA, code edit, architecture explain).
2. Add output mode contracts (brief answer, implementation plan, patch-ready notes).
3. Add confidence badges and next-action suggestions.

Acceptance:
- Outputs are consistently actionable and easy to execute.

## Usefulness rubric

The agent is useful when each run reliably gives:

1. Correct answer with evidence links.
2. Clear uncertainty when evidence is weak.
3. Concrete next steps (commands/files/tests).
4. Recoverable session behavior under failures.
