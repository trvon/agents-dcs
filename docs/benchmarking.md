# Benchmarking Guide

This project is accuracy-first. Benchmarks should prioritize correctness and reproducibility over
speed.

## Principles

- Prime YAMS indexing before retrieval so benchmarks run against current code, not stale memory.
- Keep decomposition deterministic when measuring retrieval behavior.
- Separate policy-tuned experiments from generalization checks.
- Avoid stale checkpoint carryover when run settings change.
- Record enough metadata to replay results.

## Retrieval Benchmark (No Generation)

Use this to evaluate retrieval quality (hit/recall/MRR/noise/graph utility).

```bash
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --decompose-mode heuristic \
  --task-seeding=false \
  --out results/retrieval_baseline.json
```

Default behavior:

- The command re-indexes the scoped repo with `yams add . --recursive ...` before benchmarking.
- It waits for `yams status --json` to report ready and for post-ingest queues to drain.
- Pass `--no-prime-index` only when you intentionally want to reuse a pre-ingested dataset.

Recommended settings:

- `--decompose-mode heuristic`: deterministic query planning.
- `--task-seeding=false`: disables hardcoded task-family seeds to reduce benchmark leakage.
- `--tag-match all`: require all tags when filtering.
- Run retrieval benchmark before coverage benchmark when tuning traversal or query policy.

Interpretation guidance:

- `file_hit_at_3` and `file_recall_at_5` matter more than raw symbol hit when deciding whether traversal is actually finding authoritative files.
- `graph_useful_neighbor_ratio` should go up only when graph refinement is helping; high query count with low graph utility is a smell.
- Watch `source_count` and `chunk_count` for breadth explosions that bloat downstream prompts.

If you explicitly benchmark model-driven decomposition, set:

```bash
research-agent-benchmark-retrieval --decompose-mode model --decomposer-temperature 0
```

## Coverage Benchmark (Full Pipeline)

Use this for end-to-end answer quality and routing behavior.

```bash
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models gpt-oss-120b-executor \
  --critic gpt-oss-20b-critic \
  --task-seeding=false \
  --checkpoint results/coverage_checkpoint.json \
  --out results/coverage_results.json
```

Recommended workflow:

1. Validate retrieval quality first with `research-agent-benchmark-retrieval`.
2. Run one executor stack at a time for large local models.
3. Save each run to a separate JSON file.
4. Merge runs with `research-agent-benchmark-report`.

Example sequential comparison:

```bash
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models gpt-oss-120b-executor \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/gpt120b.json

research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen-122b-executor \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/qwen122b.json

research-agent-benchmark-report results/gpt120b.json results/qwen122b.json --out results/head_to_head.json
```

Checkpoint behavior:

- Checkpoint keys include a config fingerprint (model + thresholds + filters + policy flags).
- This prevents accidental reuse of old task results under new settings.

## Scope and Paths

- Override retrieval scope with `YAMS_CWD=/path/to/repo` or `--yams-cwd`.
- Default scope resolves dynamically from the current checkout.
- Benchmarks prime indexing inside the scoped repo, so `--yams-cwd` also controls what gets re-indexed.

## Plan Review Benchmarking Readiness

Runtime plan review is available now through `research-agent review`.
When benchmarking plan-review quality, keep it separate from retrieval-only and answer-generation runs:

- retrieval benchmark: `task -> retrieval quality`
- coverage benchmark: `task -> answer quality`
- plan review benchmark: `plan + code changes -> review quality`

Do not collapse these into one score if you want meaningful model comparisons.

## Report Checklist

When publishing benchmark outputs, include:

- command line used
- models (executor/critic/fallback)
- whether indexing was primed or intentionally skipped
- decomposition mode and task-seeding mode
- task filters (`task_type`, `tags`, `tag_match`)
- output JSON path and checkpoint path
