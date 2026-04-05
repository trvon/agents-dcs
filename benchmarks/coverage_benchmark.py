from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dcs.indexing import prime_yams_index
from dcs.lmstudio_context import get_context_length, preload_model
from dcs.pipeline import DCSPipeline
from dcs.plan_review import PlanReviewer, _looks_like_rich_plan_prompt
from dcs.runtime_config import load_runtime_settings
from dcs.router import RoutingPolicy, TieredRouter
from dcs.types import (
    EvalResult,
    EvalTask,
    ModelConfig,
    PipelineConfig,
    PipelineResult,
    PlanReviewInput,
    TaskType,
)
from eval.metrics import evaluate_task
from eval.runner import EvalRunner


def _default_paths() -> tuple[str, str, str]:
    base_dir = Path(__file__).resolve().parents[1]
    runtime = load_runtime_settings(base_dir)
    task_dir = str(runtime.task_dir or (base_dir / "eval" / "tasks"))
    config_dir = runtime.config_dir or (base_dir / "configs")
    models_cfg = str(config_dir / "models.yaml")
    env_cwd = os.environ.get("YAMS_CWD", "").strip()
    if env_cwd:
        return task_dir, models_cfg, env_cwd
    if runtime.yams_cwd is not None:
        return task_dir, models_cfg, str(runtime.yams_cwd)
    # external/agent -> external -> yams
    repo_root = base_dir.parents[1]
    return task_dir, models_cfg, str(repo_root)


def _load_models_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid models config: {path}")
    return raw


def _build_model_config(models_cfg: dict[str, Any], key: str) -> ModelConfig:
    models = models_cfg.get("models") or {}
    backends = models_cfg.get("backends") or {}
    if key not in models:
        raise KeyError(f"Unknown model key: {key}")

    spec = models[key] or {}
    backend = backends.get(spec.get("backend"), {})

    name = str(spec.get("name", ""))
    suffix = spec.get("system_suffix") or ""
    if not suffix and "thinking" in name:
        suffix = "/no_think"

    return ModelConfig(
        name=name,
        base_url=str(backend.get("base_url", "http://localhost:1234/v1")),
        api_key=str(backend.get("api_key", "lm-studio")),
        context_window=int(spec.get("context_window", 8192)),
        max_output_tokens=int(spec.get("max_output_tokens", 2048)),
        temperature=float(spec.get("temperature", 0.7)),
        system_suffix=str(suffix),
        request_timeout_s=float(spec.get("request_timeout_s", 600.0)),
        max_retries=int(spec.get("max_retries", 2)),
        retry_backoff_s=float(spec.get("retry_backoff_s", 2.0)),
    )


def _build_model_config_or_id(
    models_cfg: dict[str, Any],
    raw: str,
    *,
    preferred_role: str | None = None,
    default_temperature: float = 0.0,
) -> ModelConfig:
    models = models_cfg.get("models") or {}
    if raw in models:
        return _build_model_config(models_cfg, raw)

    for key, spec in models.items():
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "")
        if name != raw:
            continue
        role = str(spec.get("role") or "")
        if preferred_role is None or role == preferred_role:
            return _build_model_config(models_cfg, str(key))

    if not raw.startswith("openai/"):
        openai_name = f"openai/{raw}"
        for key, spec in models.items():
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name") or "")
            if name != openai_name:
                continue
            role = str(spec.get("role") or "")
            if preferred_role is None or role == preferred_role:
                return _build_model_config(models_cfg, str(key))

    backends = models_cfg.get("backends") or {}
    default_backend = backends.get("lmstudio") or {}
    suffix = "/no_think" if "thinking" in str(raw).lower() else ""
    return ModelConfig(
        name=str(raw),
        base_url=str(default_backend.get("base_url", "http://localhost:1234/v1")),
        api_key=str(default_backend.get("api_key", "lm-studio")),
        context_window=8192,
        max_output_tokens=1024,
        temperature=float(default_temperature),
        system_suffix=suffix,
        request_timeout_s=600.0,
        max_retries=2,
        retry_backoff_s=2.0,
    )


def _collect_sources(pr: PipelineResult | None) -> list[str]:
    if not pr:
        return []
    sources: set[str] = set()
    for it in pr.iterations:
        if it.context:
            sources.update([s for s in it.context.sources if s])
        for qr in it.query_results:
            for chunk in qr.chunks:
                if chunk.source:
                    sources.add(chunk.source)
    return sorted(sources)


def _filter_by_tags(tasks: list[EvalTask], tags: set[str], *, mode: str = "all") -> list[EvalTask]:
    if not tags:
        return tasks
    out: list[EvalTask] = []
    match_all = str(mode or "all").lower() != "any"
    for t in tasks:
        ttags = {str(x).strip().lower() for x in (t.tags or []) if str(x).strip()}
        ok = tags.issubset(ttags) if match_all else bool(tags.intersection(ttags))
        if ok:
            out.append(t)
    return out


def _checkpoint_key(name: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}"


def _task_type_from_str(raw: str | None) -> TaskType | None:
    if not raw:
        return None
    s = raw.strip().lower()
    for t in TaskType:
        if t.value == s:
            return t
    return None


def _preload_configs(
    console: Console,
    model_cfgs: list[ModelConfig],
    *,
    retries: int,
    retry_backoff_s: float,
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for cfg in model_cfgs:
        key = (cfg.name, cfg.base_url, cfg.api_key)
        if key in seen:
            continue
        seen.add(key)
        requested_ctx = int(cfg.context_window)
        ok = preload_model(
            cfg.name,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            context_length=requested_ctx,
            min_ready_context_length=65535,
            keep_model_in_memory=True,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            ready_timeout_s=max(300.0, float(cfg.request_timeout_s or 600.0)),
            ready_poll_s=max(1.0, float(retry_backoff_s)),
            required_successes=2,
        )
        actual_ctx = get_context_length(cfg.name) or int(cfg.context_window)
        try:
            cfg.context_window = int(actual_ctx)
        except Exception:
            pass

        status = "ok" if ok else "failed"
        console.print(
            "[dim]Preload "
            f"{status}: {cfg.name} (requested_ctx={requested_ctx} actual_ctx={int(actual_ctx)})"
            "[/dim]"
        )


def _disable_heavy_secondary_calls(config: PipelineConfig) -> PipelineConfig:
    config.use_dspy_faithfulness = False
    config.no_ground_truth_mode = False
    return config


def _task_plan_text(task: EvalTask) -> str:
    plan = str(getattr(task, "plan", "") or "").strip()
    if plan:
        return plan
    if _looks_like_rich_plan_prompt(task.description):
        return task.description.strip()
    return ""


async def _maybe_attach_plan_review(
    config: PipelineConfig,
    task: EvalTask,
    pipeline_result: PipelineResult | None,
    *,
    enabled: bool,
) -> PipelineResult | None:
    if not enabled or pipeline_result is None:
        return pipeline_result

    plan_text = _task_plan_text(task)
    if not plan_text:
        return pipeline_result

    reviewer = PlanReviewer(config)
    review = await reviewer.review(
        PlanReviewInput(
            plan=plan_text,
            task=task.description,
            execution_summary=pipeline_result.final_output,
            changed_files=_collect_sources(pipeline_result)[
                : int(config.plan_review_max_changed_files or 8)
            ],
        )
    )
    pipeline_result.plan_review = review
    return pipeline_result


async def _run_suite(
    config: PipelineConfig,
    tasks: list[EvalTask],
    *,
    enable_plan_review: bool = False,
    checkpoint: dict[str, list[dict[str, Any]]] | None = None,
    checkpoint_key: str | None = None,
    checkpoint_path: Path | None = None,
    fallback_configs: list[PipelineConfig] | None = None,
    fallback_threshold: float | None = None,
) -> list[EvalResult]:
    runner = EvalRunner(config, task_dir=".")
    results: list[EvalResult] = []
    for task in tasks:
        try:
            if fallback_configs:
                threshold = (
                    float(fallback_threshold)
                    if fallback_threshold is not None
                    else float(config.quality_threshold)
                )
                router = TieredRouter(
                    base_config=config,
                    fallback_configs=fallback_configs,
                    policy=RoutingPolicy(
                        quality_threshold=threshold,
                        min_sources=1,
                        require_non_error_output=True,
                        preload_tier_models=True,
                        preload_retries=2,
                        preload_retry_backoff_s=2.0,
                    ),
                )
                routed = await router.run(task.description)
                pr = routed.selected_result
                selected_tier = routed.selected_tier
                escalated = 1.0 if routed.escalated else 0.0
            else:
                pipe = DCSPipeline(config)
                pr = await pipe.run(task.description)
                selected_tier = 0
                escalated = 0.0

            pr = await _maybe_attach_plan_review(
                config,
                task,
                pr,
                enabled=enable_plan_review,
            )

            metrics = evaluate_task(task, pr)
            sources = _collect_sources(pr)
            metrics["source_count"] = float(len(sources))
            metrics["escalated"] = escalated
            metrics["selected_tier"] = float(selected_tier)
            passed = runner._decide_pass(task, metrics)
            result = EvalResult(task_id=task.id, pipeline_result=pr, metrics=metrics, passed=passed)

            results.append(result)
        except Exception as e:
            results.append(
                EvalResult(
                    task_id=task.id, pipeline_result=None, metrics={}, passed=False, error=str(e)
                )
            )

        if checkpoint is not None and checkpoint_key and checkpoint_path:
            payload = _serialize_results(results)
            existing = checkpoint.get(checkpoint_key, [])
            existing_ids = {str(x.get("task_id")) for x in existing}
            for item in payload:
                if str(item.get("task_id")) not in existing_ids:
                    existing.append(item)
                    existing_ids.add(str(item.get("task_id")))
            checkpoint[checkpoint_key] = existing
            _write_checkpoint(checkpoint_path, checkpoint)
    return results


def _render_table(console: Console, title: str, results: list[EvalResult]) -> None:
    tbl = Table(title=title)
    tbl.add_column("Task")
    tbl.add_column("Pass", justify="center")
    tbl.add_column("Quality", justify="right")
    tbl.add_column("Sources", justify="right")
    tbl.add_column("Latency ms", justify="right")
    tbl.add_column("Iters", justify="right")

    for r in results:
        q = r.metrics.get("quality_score", 0.0)
        lat = r.metrics.get("total_latency_ms", 0.0)
        iters = r.metrics.get("iterations", 0.0)
        sources = r.metrics.get("source_count", 0.0)
        tbl.add_row(
            r.task_id,
            "yes" if r.passed else "no",
            f"{q:.2f}",
            f"{sources:.0f}",
            f"{lat:.0f}",
            f"{iters:.0f}",
        )
    console.print(tbl)


def _serialize_results(results: list[EvalResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        payload: dict[str, Any] = {
            "task_id": r.task_id,
            "task_type": r.task_type,
            "tags": list(r.tags or []),
            "repeat_index": int(r.repeat_index or 1),
            "passed": r.passed,
            "metrics": r.metrics,
            "error": r.error,
        }
        if r.pipeline_result:
            payload["pipeline"] = {
                "task": r.pipeline_result.task,
                "iterations": r.pipeline_result.num_iterations,
                "converged": r.pipeline_result.converged,
                "total_latency_ms": r.pipeline_result.total_latency_ms,
            }
            if r.pipeline_result.plan_review is not None:
                payload["pipeline"]["plan_review"] = {
                    "coverage_score": float(r.pipeline_result.plan_review.coverage_score or 0.0),
                    "executed_well": bool(r.pipeline_result.plan_review.executed_well),
                    "summary": str(r.pipeline_result.plan_review.summary or ""),
                }
            payload["sources"] = _collect_sources(r.pipeline_result)
        out.append(payload)
    return out


def _load_checkpoint(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, val in raw.items():
        if isinstance(val, list):
            out[key] = [v for v in val if isinstance(v, dict)]
    return out


def _write_checkpoint(path: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _result_from_dict(d: dict[str, Any]) -> EvalResult:
    return EvalResult(
        task_id=str(d.get("task_id", "")),
        pipeline_result=None,
        metrics=d.get("metrics") or {},
        passed=bool(d.get("passed", False)),
        error=d.get("error"),
        task_type=str(d.get("task_type", "") or ""),
        tags=[str(x) for x in (d.get("tags") or []) if x is not None],
        repeat_index=int(d.get("repeat_index") or 1),
    )


def main() -> int:
    default_task_dir, default_models_cfg, default_yams_cwd = _default_paths()

    parser = argparse.ArgumentParser(description="Run coverage benchmark suite over eval tasks")
    parser.add_argument("--task-dir", default=default_task_dir, help="Task directory")
    parser.add_argument("--task-type", default=None, help="Task type filter (qa|coding)")
    parser.add_argument("--tags", default="", help="Comma-separated tag filter")
    parser.add_argument(
        "--tag-match",
        choices=["all", "any"],
        default="all",
        help="Tag matching mode (all tags required or any tag)",
    )
    parser.add_argument("--models", default="", help="Comma-separated executor model keys")
    parser.add_argument("--critic", default=None, help="Critic model key (or 'same')")
    parser.add_argument("--context-budget", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--convergence-delta", type=float, default=0.05)
    parser.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default="auto",
        help="Context budget profile selection",
    )
    parser.add_argument("--yams-cwd", default=default_yams_cwd)
    parser.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    parser.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    parser.add_argument("--models-config", default=default_models_cfg)
    parser.add_argument(
        "--task-seeding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable task-specific hardcoded query seeds in decomposition",
    )
    parser.add_argument("--out", default=None, help="Write JSON results to path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint JSON path")
    parser.add_argument("--batch-size", type=int, default=0, help="Run at most N pending tasks")
    parser.add_argument(
        "--fallback-models", default="", help="Comma-separated fallback executor keys"
    )
    parser.add_argument("--fallback-critic", default=None, help="Critic model key for fallbacks")
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=None,
        help="Quality threshold to trigger fallback",
    )
    parser.add_argument(
        "--preload-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload/pin executor and fallback models before each suite",
    )
    parser.add_argument(
        "--preload-retries",
        type=int,
        default=3,
        help="Retries for model preload warmup",
    )
    parser.add_argument(
        "--preload-retry-backoff-s",
        type=float,
        default=2.0,
        help="Backoff seconds between preload retries",
    )
    parser.add_argument(
        "--codemap-budget",
        type=int,
        default=None,
        help="Override codemap token budget (0 disables codemap)",
    )
    parser.add_argument(
        "--prime-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-index repo content and wait for ingestion to settle before running coverage suite",
    )
    parser.add_argument(
        "--prime-timeout-s",
        type=float,
        default=900.0,
        help="Timeout for pre-benchmark indexing wait",
    )
    parser.add_argument(
        "--plan-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run plan review for tasks that provide a plan or rich plan prompt",
    )
    args = parser.parse_args()

    console = Console()
    models_cfg = _load_models_config(Path(args.models_config))
    defaults = models_cfg.get("defaults") or {}

    if args.prime_index:
        console.print(f"Priming YAMS index under {args.yams_cwd} ...")
        status = prime_yams_index(
            root=str(args.yams_cwd),
            timeout_s=float(args.prime_timeout_s),
        )
        post_ingest = status.get("post_ingest") if isinstance(status, dict) else {}
        queued = post_ingest.get("queued", 0) if isinstance(post_ingest, dict) else 0
        prime_meta = status.get("_dcs_prime") if isinstance(status, dict) else {}
        reason = prime_meta.get("reason", "ready") if isinstance(prime_meta, dict) else "ready"
        skipped = (
            bool(prime_meta.get("skipped_add", False)) if isinstance(prime_meta, dict) else False
        )
        console.print(
            f"YAMS ready; post_ingest queued={queued} reason={reason} skipped_add={'yes' if skipped else 'no'}"
        )

    executors = [m for m in args.models.split(",") if m.strip()]
    if not executors:
        executors = [defaults.get("executor", "qwen3-4b")]

    critic_key = args.critic or defaults.get("critic", "devstral")
    fallback_keys = [m for m in args.fallback_models.split(",") if m.strip()]
    fallback_critic_key = args.fallback_critic or critic_key

    runner = EvalRunner(PipelineConfig(), task_dir=args.task_dir)
    tasks = runner.load_tasks(args.task_dir, task_type=_task_type_from_str(args.task_type))
    tags = {t.strip().lower() for t in args.tags.split(",") if t.strip()}
    tasks = _filter_by_tags(tasks, tags, mode=str(args.tag_match))

    if not tasks:
        console.print("No tasks found for selection")
        return 1

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_data = _load_checkpoint(checkpoint_path) if checkpoint_path else {}

    all_results: dict[str, list[EvalResult]] = {}

    for exec_key in executors:
        executor_cfg = _build_model_config_or_id(
            models_cfg, exec_key, preferred_role="executor", default_temperature=1.0
        )
        if critic_key == "same":
            critic_cfg = executor_cfg
        else:
            critic_cfg = _build_model_config_or_id(
                models_cfg, critic_key, preferred_role="critic", default_temperature=0.0
            )

        config = PipelineConfig(
            executor_model=executor_cfg,
            critic_model=critic_cfg,
            context_budget=args.context_budget,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            convergence_delta=args.convergence_delta,
            context_profile=str(args.context_profile),
            no_ground_truth_mode=bool(args.ground_truth_mode),
            use_dspy_faithfulness=bool(args.dspy_faithfulness),
            enable_task_seeding=bool(args.task_seeding),
            yams_cwd=args.yams_cwd,
            codemap_budget=(
                int(args.codemap_budget)
                if args.codemap_budget is not None
                else PipelineConfig().codemap_budget
            ),
            codemap_max_files=5,
            codemap_max_symbols_per_file=8,
            codemap_include_type_counts=False,
        )
        config = _disable_heavy_secondary_calls(config)

        fallback_configs: list[PipelineConfig] = []
        if fallback_keys:
            for fb_key in fallback_keys:
                fb_exec = _build_model_config_or_id(
                    models_cfg, fb_key, preferred_role="executor", default_temperature=1.0
                )
                fb_crit = (
                    fb_exec
                    if fallback_critic_key == "same"
                    else _build_model_config_or_id(
                        models_cfg,
                        fallback_critic_key,
                        preferred_role="critic",
                        default_temperature=0.0,
                    )
                )
                fallback_configs.append(
                    _disable_heavy_secondary_calls(
                        PipelineConfig(
                            executor_model=fb_exec,
                            critic_model=fb_crit,
                            context_budget=args.context_budget,
                            max_iterations=args.max_iterations,
                            quality_threshold=args.quality_threshold,
                            convergence_delta=args.convergence_delta,
                            context_profile=str(args.context_profile),
                            no_ground_truth_mode=bool(args.ground_truth_mode),
                            use_dspy_faithfulness=bool(args.dspy_faithfulness),
                            enable_task_seeding=bool(args.task_seeding),
                            yams_cwd=args.yams_cwd,
                            codemap_budget=(
                                int(args.codemap_budget)
                                if args.codemap_budget is not None
                                else PipelineConfig().codemap_budget
                            ),
                            codemap_max_files=5,
                            codemap_max_symbols_per_file=8,
                            codemap_include_type_counts=False,
                        )
                    )
                )

        if args.preload_models:
            preload_list = [config.executor_model]
            if config.critic_model:
                preload_list.append(config.critic_model)
            for fb in fallback_configs:
                preload_list.append(fb.executor_model)
                if fb.critic_model:
                    preload_list.append(fb.critic_model)
            _preload_configs(
                console,
                preload_list,
                retries=args.preload_retries,
                retry_backoff_s=args.preload_retry_backoff_s,
            )

        checkpoint_payload = {
            "executor": exec_key,
            "critic": critic_key,
            "fallback_models": sorted(fallback_keys),
            "fallback_critic": str(fallback_critic_key),
            "task_dir": str(args.task_dir),
            "task_type": str(args.task_type),
            "tags": sorted(tags),
            "tag_match": str(args.tag_match),
            "task_count": int(len(tasks)),
            "context_budget": int(args.context_budget),
            "max_iterations": int(args.max_iterations),
            "quality_threshold": float(args.quality_threshold),
            "convergence_delta": float(args.convergence_delta),
            "context_profile": str(args.context_profile),
            "ground_truth_mode": bool(args.ground_truth_mode),
            "dspy_faithfulness": bool(args.dspy_faithfulness),
            "task_seeding": bool(args.task_seeding),
            "fallback_threshold": args.fallback_threshold,
            "preload_models": bool(args.preload_models),
            "prime_index": bool(args.prime_index),
            "codemap_budget": (
                int(args.codemap_budget) if args.codemap_budget is not None else None
            ),
        }
        ckpt_key = _checkpoint_key(exec_key, checkpoint_payload)

        console.print(f"\n=== Running {exec_key} (critic={critic_key}) ===")

        existing = checkpoint_data.get(ckpt_key, [])
        if not existing:
            # Backward compatibility with older checkpoint keying.
            existing = checkpoint_data.get(exec_key, [])
        done_ids = {str(x.get("task_id")) for x in existing}
        pending = [t for t in tasks if t.id not in done_ids]
        if args.batch_size and args.batch_size > 0:
            pending = pending[: args.batch_size]

        results = asyncio.run(
            _run_suite(
                config,
                pending,
                enable_plan_review=bool(args.plan_review),
                checkpoint=checkpoint_data,
                checkpoint_key=ckpt_key,
                checkpoint_path=checkpoint_path,
                fallback_configs=fallback_configs,
                fallback_threshold=args.fallback_threshold,
            )
        )

        new_payload = _serialize_results(results)
        new_ids = {r.task_id for r in results}
        combined = [*new_payload, *[x for x in existing if str(x.get("task_id")) not in new_ids]]
        all_results[exec_key] = [_result_from_dict(x) for x in combined if isinstance(x, dict)]
        _render_table(console, f"Results: {exec_key}", all_results[exec_key])

    if args.out:
        payload = {k: _serialize_results(v) for k, v in all_results.items()}
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
