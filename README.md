# Dynamic Context Scaffold (DCS)

[![CI](https://github.com/trvon/agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/trvon/agent/actions/workflows/ci.yml)

YAMS-backed retrieval, evaluation, and plan-review harness for local and hosted LLM agents.

Thesis: scaffold quality can close part of the gap between smaller models and larger models on
knowledge-intensive tasks.

## Architecture

```
task -> prime index -> decompose -> search -> validate anchors -> graph refine -> targeted get/grep
     -> assemble -> execute (LM Studio) -> critique -> iterate

plan + code changes -> ingest review artifacts -> retrieve repo context -> review -> gaps/tests/follow-up
```

## Quick Start

```bash
uv sync
research-agent status
research-agent run "Explain how YAMS hybrid search works"

# Eval suite
research-agent eval --task-dir eval/tasks --type coding

# Scaffolded vs vanilla comparison
research-agent compare --task-dir eval/tasks

# Post-execution plan review
research-agent review --task "add plan review mode" --plan-file plan.txt --change-summary-file changes.txt

# Retrieval benchmark smoke check
research-agent-benchmark-retrieval --task-dir eval/tasks --out results/retrieval_smoke.json

# Merge sequential benchmark runs from separately loaded models
research-agent-benchmark-report results/qwen122b.json results/qwen35b.json --out results/head_to_head.json
```

`dcs` remains available as a compatibility alias.

## Runtime Notes

- Executor configs default to `temperature: 1.0`; critic and review configs default to `temperature: 0.0`.
- Runtime path/debug defaults can live in `config.toml`; env vars remain available as ad hoc overrides.
- Benchmark commands prime YAMS indexing by default, wait for post-ingest queues to settle, then run retrieval.
- Benchmark priming now uses `yams add --sync` instead of repeated status polling, so the harness waits once instead of spamming CLI/status checks.
- Model warmup now waits for repeated successful probes and at least `65535` tokens of ready context before long benchmark runs continue.
- If the indexed repo fingerprint is unchanged and YAMS is already ready, benchmark priming skips the re-add/re-wait cycle.
- Large local models should be benchmarked one at a time, then merged with `research-agent-benchmark-report`.

## Benchmarking

Use benchmark commands for reproducible measurement, not ad-hoc runs.

```bash
# Retrieval-only benchmark (deterministic decomposition by default)
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --decompose-mode heuristic \
  --task-seeding=false \
  --out results/retrieval_baseline.json

# Coverage benchmark (full pipeline)
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen-122b-executor \
  --critic gpt-oss-20b-critic \
  --plan-review \
  --task-seeding=false \
  --checkpoint results/coverage_checkpoint.json \
  --out results/coverage_results.json
```

Recommended workflow:

1. Run retrieval-only benchmark first to verify file targeting and graph utility.
2. Run full coverage benchmark after retrieval looks healthy.
3. For large-model comparisons, run each model in a separate pass and save separate JSON outputs.
4. Merge the saved outputs with `research-agent-benchmark-report`.

Sequential large-model example:

```bash
# Qwen 122B executor + GPT OSS 20B critic
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen-122b-executor \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/qwen122b_vs_gpt20critic.json

# Qwen 35B A3B executor + GPT OSS 20B critic
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen35-35b-a3b \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/qwen35b_vs_gpt20critic.json

# Merge the separate runs
research-agent-benchmark-report \
  results/qwen122b_vs_gpt20critic.json \
  results/qwen35b_vs_gpt20critic.json \
  --out results/head_to_head.json
```

Notes:
- Benchmark checkpoint keys now include a config fingerprint to avoid stale-result reuse.
- Set `YAMS_CWD` if you want to override repository scope.
- Use `--task-seeding=false` for fairer generalization measurements.
- For large local models, run benchmarks one model at a time and merge the JSON outputs with `research-agent-benchmark-report`.
- Benchmarks prime YAMS indexing by default so retrieval runs against freshly ingested code before searching.
- Pass `--no-prime-index` only if you have already indexed and intentionally want to skip the pre-run ingest step.
- Current mixed-suite retrieval baseline is the non-DSPy rerank result in `results/retrieval_rerank_v2.json`.
- DSPy retrieval rerank and optimizer flags are kept for experiments, but they are not the recommended default path.

See `docs/benchmarking.md` for the full protocol.

## Plan Review

Use plan review after a coding agent finishes work and you want a second pass on execution quality.

```bash
research-agent review \
  --task "Add plan review mode and benchmark reporting" \
  --plan-file plan.txt \
  --change-summary-file changes.txt \
  --execution-summary-file execution.txt \
  --changed-files "dcs/plan_review.py,dcs/cli.py,benchmarks/report_benchmark.py" \
  --json-out results/plan_review.json
```

The reviewer ingests compact plan/change artifacts into YAMS, retrieves supporting repo context, and returns:

- step coverage
- gaps and risks
- suggested tests
- follow-up implementation advice

If `--task` itself is a structured research/build prompt with sections like `Tests First`, `Acceptance Gates`, or `Benchmark Cases To Rerun`, the reviewer can use that prompt directly as the plan source.

## Context Profiles

- `--context-profile auto` (default): switches to larger budgets when model context is large.
- `--context-profile standard`: baseline budgets.
- `--context-profile large`: force larger retrieval/system/output/codemap budgets.

At run start, DCS prints requested vs actual context window for executor/critic.

## Requirements

- YAMS daemon available (`yams serve` for MCP)
- LM Studio with models loaded (OpenAI-compatible API at `http://localhost:1234/v1`)
- Python 3.11+

## Repo-Local Hooks (Optional)

- `pre-commit`: `ruff` lint/fix/format on staged Python files
- `pre-push`: `pytest -q`

Enable once (inside `external/agent/`):

```bash
git config core.hooksPath .githooks
```

## CI

GitHub Actions runs lint, format checks, tests with coverage, and package builds on push/PR:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -q --cov=dcs.cli --cov=dcs.router --cov=dcs.types --cov=eval.metrics --cov=eval.runner --cov-report=term-missing --cov-report=xml:coverage-core.xml --cov-fail-under=80`
- `uv run pytest -q --cov=dcs --cov=benchmarks --cov=eval --cov-report=term --cov-report=xml:coverage-full.xml` (snapshot, non-gating)
- `uv build`

Coverage XML artifacts are uploaded per Python version.
Current CI coverage floor: 80% for the core control-plane modules.

### Branch Protection (Recommended)

Require these status checks on `main`:

- `lint-test-build (3.11)`
- `lint-test-build (3.12)`
