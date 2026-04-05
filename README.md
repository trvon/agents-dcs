# Dynamic Context Scaffold (DCS)

[![CI](https://github.com/trvon/agents-dcs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/trvon/agents-dcs/actions/workflows/ci.yml)

DCS is a YAMS-backed retrieval, evaluation, and plan-review harness for local and hosted LLM agents. The repo is accuracy-first: retrieval coverage and grounded outputs matter more than speed.

## What It Does

- decomposes tasks into targeted retrieval queries
- retrieves code and graph context from YAMS
- assembles bounded context for an executor model
- critiques output quality and faithfulness
- benchmarks retrieval and end-to-end coverage
- reviews completed implementation work against a plan

## Quick Start

```bash
uv sync --dev
research-agent status
research-agent run "Explain how YAMS hybrid search works"
```

## Core Commands

```bash
# Evaluate scaffolded runs against the task suite
research-agent eval --task-dir eval/tasks --type coding

# Compare scaffolded vs vanilla runs
research-agent compare --task-dir eval/tasks

# Review completed work against a plan
research-agent review \
  --task "add plan review mode" \
  --plan-file plan.txt \
  --change-summary-file changes.txt

# Retrieval benchmark smoke check
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --out results/retrieval_smoke.json
```

`dcs` remains available as a compatibility alias.

## Configuration

Runtime settings are optional. DCS looks for `config.toml` in this order:

1. `DCS_CONFIG_TOML`
2. `<repo>/config.toml`
3. `<cwd>/config.toml`

Supported settings today:

```toml
[paths]
config_dir = "configs"
task_dir = "eval/tasks"
yams_cwd = "../.."

[debug]
critic_debug_dir = ""
```

Environment variables still work as ad hoc overrides.

The checked-in `config.example.toml` documents the supported fields. A local `config.toml` is intended for checkout-specific overrides and is gitignored.

## Benchmarking

Use benchmark commands for reproducible measurement instead of ad hoc runs.

```bash
# Retrieval-only benchmark
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --decompose-mode heuristic \
  --task-seeding=false \
  --out results/retrieval_baseline.json

# Full coverage benchmark
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

1. Run retrieval-only benchmark first.
2. Run full coverage benchmark after retrieval quality looks healthy.
3. Run large local models one at a time.
4. Merge separate outputs with `research-agent-benchmark-report`.

Full benchmark protocol lives in `docs/benchmarking.md`.

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

The reviewer ingests compact plan and change artifacts into YAMS, retrieves supporting repo context, and returns:

- step coverage
- gaps and risks
- suggested tests
- follow-up implementation advice

## Runtime Notes

- Executor configs default to `temperature: 1.0`; critic and review configs default to `temperature: 0.0`.
- Benchmark commands prime YAMS indexing by default and wait for ingest to settle before retrieval.
- Large local models should usually be benchmarked one at a time and merged later.
- If the indexed repo fingerprint is unchanged and YAMS is already ready, benchmark priming skips unnecessary re-ingest work.

## Requirements

- YAMS daemon available for MCP usage
- LM Studio available at `http://localhost:1234/v1`
- Python 3.11+

## Repo-Local Hooks

- `pre-commit`: `ruff` lint/fix/format on staged Python files
- `pre-push`: `pytest -q`

Enable once inside this repo:

```bash
git config core.hooksPath .githooks
```

## CI

GitHub Actions runs on Python 3.11 and 3.12 and checks:

- `uv run ruff check .`
- `uv run ruff format --check .`
- gated core-module coverage
- full non-gating coverage snapshot
- `uv build`

## Docs

- `docs/benchmarking.md`: benchmark protocol and interpretation
- `docs/agent-architecture-v2.md`: architecture direction
- `docs/execution-plan.md`: implementation roadmap
