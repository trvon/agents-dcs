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
- It uses `yams add --sync` so indexing and extraction finish in one bounded wait instead of repeated status polling.
- Model warmup waits for repeated successful probes and at least `65535` ready-context tokens before the suite continues.
- If the repo fingerprint is unchanged and YAMS is already ready, priming skips the re-index step and does not add unnecessary settle time.
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

Current retrieval baseline:

- Keep the non-DSPy file-authority rerank baseline from `results/retrieval_rerank_v2.json` as the mixed-suite reference.
- DSPy retrieval reranking and BootstrapFewShot optimizer paths remain experimental; they improved noise/latency in some runs but did not beat the baseline on primary file-ranking metrics.
- If you run DSPy experiments anyway, prefer explicit raw ids for all model flags so executor, critic, and reranker can share one loaded LM Studio model.

If you explicitly benchmark model-driven decomposition, set:

```bash
research-agent-benchmark-retrieval --decompose-mode model --decomposer-temperature 0
```

## Coverage Benchmark (Full Pipeline)

Use this for end-to-end answer quality and routing behavior.

```bash
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

1. Validate retrieval quality first with `research-agent-benchmark-retrieval`.
2. Run one executor stack at a time for large local models.
3. Save each run to a separate JSON file.
4. Merge runs with `research-agent-benchmark-report`.

Example sequential comparison:

```bash
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen-122b-executor \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/qwen122b.json

research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen35-35b-a3b \
  --critic gpt-oss-20b-critic \
  --context-profile large \
  --task-seeding=false \
  --out results/qwen35b.json

research-agent-benchmark-report results/qwen122b.json results/qwen35b.json --out results/head_to_head.json
```

Plan review in coverage runs:

- Pass `--plan-review` to attach plan-review metrics during coverage benchmarking.
- Tasks can provide structured plan text via top-level `plan:` or `evaluation.plan:`.
- If no explicit plan is present but the task description is itself a structured research/build prompt, the benchmark treats that prompt as the plan source automatically.
- Report output includes `plan_coverage_mean` when plan review metrics are present.

Checkpoint behavior:

- Checkpoint keys include a config fingerprint (model + thresholds + filters + policy flags).
- This prevents accidental reuse of old task results under new settings.

## Scope and Paths

- Override retrieval scope with `YAMS_CWD=/path/to/repo` or `--yams-cwd`.
- Default scope resolves dynamically from the current checkout.
- Benchmarks prime indexing inside the scoped repo, so `--yams-cwd` also controls what gets re-indexed.
- Checked-in runtime defaults now live in `config.toml` for config/task/yams paths and critic debug artifacts.

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
