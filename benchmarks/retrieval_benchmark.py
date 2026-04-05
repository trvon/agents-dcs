from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dcs.client import YAMSClient
from dcs.decomposer import TaskDecomposer
from dcs.indexing import prime_yams_index
from dcs.planner import QueryPlanner
from dcs.runtime_config import load_runtime_settings
from dcs.types import (
    EvalTask,
    ModelConfig,
    PipelineConfig,
    QuerySpec,
    QueryType,
    TaskType,
    YAMSChunk,
)
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
        raise ValueError(f"invalid models config: {path}")
    return raw


def _build_model_config(models_cfg: dict[str, Any], key: str) -> ModelConfig:
    models = models_cfg.get("models") or {}
    backends = models_cfg.get("backends") or {}
    if key not in models:
        raise KeyError(f"model key not found: {key}")
    m = models[key] or {}
    backend = backends.get(m.get("backend"), {}) if isinstance(m, dict) else {}

    def _pick(name: str, default: Any) -> Any:
        value = m.get(name) if isinstance(m, dict) else None
        return default if value is None else value

    return ModelConfig(
        name=str(_pick("name", key)),
        base_url=str(_pick("base_url", backend.get("base_url") or "http://localhost:1234/v1")),
        api_key=str(_pick("api_key", backend.get("api_key") or "lm-studio")),
        context_window=int(_pick("context_window", 8192)),
        max_output_tokens=int(_pick("max_output_tokens", 2048)),
        temperature=float(_pick("temperature", 0.7)),
        request_timeout_s=float(_pick("request_timeout_s", 600.0)),
        max_retries=int(_pick("max_retries", 2)),
        retry_backoff_s=float(_pick("retry_backoff_s", 2.0)),
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
    return ModelConfig(
        name=str(raw),
        base_url=str(default_backend.get("base_url") or "http://localhost:1234/v1"),
        api_key=str(default_backend.get("api_key") or "lm-studio"),
        context_window=8192,
        max_output_tokens=1024,
        temperature=float(default_temperature),
        request_timeout_s=600.0,
        max_retries=2,
        retry_backoff_s=2.0,
    )


def _task_type_from_str(raw: str | None) -> TaskType | None:
    if not raw:
        return None
    s = raw.strip().lower()
    for t in TaskType:
        if t.value == s:
            return t
    return None


def _filter_by_tags(tasks: list[EvalTask], tags: set[str], *, mode: str = "all") -> list[EvalTask]:
    if not tags:
        return tasks
    out: list[EvalTask] = []
    match_all = str(mode or "all").lower() != "any"
    for t in tasks:
        ttags = {x.strip().lower() for x in (t.tags or []) if isinstance(x, str)}
        ok = tags.issubset(ttags) if match_all else bool(tags.intersection(ttags))
        if ok:
            out.append(t)
    return out


def _heuristic_decompose(task_text: str, max_queries: int) -> list[QuerySpec]:
    """Deterministic decomposition for reproducible retrieval benchmarks."""
    decomposer = TaskDecomposer.__new__(TaskDecomposer)
    specs = decomposer._fallback_decompose(task=task_text, max_queries=max_queries)
    return list(specs or [])[: max(1, int(max_queries))]


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").strip().lower()


def _match_file(source: str, gold_file: str) -> bool:
    s = _norm_path(source)
    g = _norm_path(gold_file)
    if not s or not g:
        return False
    if "/" in g:
        return s.endswith(g) or g in s
    return Path(s).name == g


def _is_noise_source(path: str) -> bool:
    p = _norm_path(path)
    if not p:
        return False
    if "/tests/" in p or "/docs/" in p or "/benchmarks/" in p:
        return True
    return p.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".lock"))


def _pattern_match(text: str, pattern: str) -> bool:
    txt = text or ""
    pat = (pattern or "").strip()
    if not pat:
        return False
    if pat.startswith("re:"):
        try:
            return re.search(pat[3:], txt, flags=re.IGNORECASE) is not None
        except re.error:
            return pat[3:].lower() in txt.lower()
    return pat.lower() in txt.lower()


def _ranked_chunks(results: list[Any]) -> list[tuple[float, QueryType, YAMSChunk]]:
    scored: list[tuple[float, QueryType, YAMSChunk]] = []
    for res in results:
        spec = res.spec
        chunks = sorted(list(res.chunks or []), key=lambda c: float(c.score or 0.0), reverse=True)
        for i, c in enumerate(chunks):
            rank = (
                (0.72 * float(c.score or 0.0)) + (0.28 * float(spec.importance or 0.0)) - (0.01 * i)
            )
            scored.append((rank, spec.query_type, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _unique_sources(scored_chunks: list[tuple[float, QueryType, YAMSChunk]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for _, _, c in scored_chunks:
        s = (c.source or "").strip()
        if not s:
            continue
        sn = _norm_path(s)
        if sn in seen:
            continue
        seen.add(sn)
        out.append(s)
    return out


def _collect_symbols(task: EvalTask) -> list[str]:
    gt = task.ground_truth or {}
    out: list[str] = []
    for key in ("symbols", "patterns"):
        vals = gt.get(key)
        if isinstance(vals, list):
            for x in vals:
                s = str(x).strip()
                if s and s not in out:
                    out.append(s)
    return out


def _collect_files(task: EvalTask) -> list[str]:
    gt = task.ground_truth or {}
    vals = gt.get("files")
    if not isinstance(vals, list):
        return []
    out: list[str] = []
    for x in vals:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def _demo_terms(task: EvalTask) -> list[str]:
    gt = task.ground_truth or {}
    out: list[str] = []
    for key in ("symbols", "patterns"):
        vals = gt.get(key)
        if not isinstance(vals, list):
            continue
        for item in vals:
            token = str(item).strip()
            if token.startswith("re:"):
                token = token[3:]
            token = re.sub(r"[^A-Za-z0-9_./-]+", " ", token).strip()
            if token and token not in out:
                out.append(token)
    return out


def _select_demo_tasks(tasks: list[EvalTask], current_task: EvalTask, limit: int) -> list[EvalTask]:
    if limit <= 0:
        return []
    buckets: dict[str, list[EvalTask]] = {}
    for task in tasks:
        if task.id == current_task.id:
            continue
        buckets.setdefault(task.task_type.value, []).append(task)

    ordered_types = sorted(buckets)
    selected: list[EvalTask] = []
    while len(selected) < limit:
        progressed = False
        for task_type in ordered_types:
            items = buckets.get(task_type) or []
            if not items:
                continue
            selected.append(items.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _build_dspy_rerank_demos(
    tasks: list[EvalTask], current_task: EvalTask, limit: int
) -> list[dict[str, Any]]:
    selected = _select_demo_tasks(tasks, current_task, limit)
    if not selected:
        return []

    all_other_tasks = [task for task in tasks if task.id != current_task.id]
    demos: list[dict[str, Any]] = []
    for task in selected:
        positives = _collect_files(task)[:2]
        if not positives:
            continue

        distractors: list[tuple[str, EvalTask | None]] = []
        for other in all_other_tasks:
            if other.id == task.id:
                continue
            for path in _collect_files(other):
                if path not in positives:
                    distractors.append((path, other))
                    break
            if len(distractors) >= 2:
                break
        while len(distractors) < 2:
            fallback = [
                ("docs/architecture.md", None),
                ("tests/unit/example_test.cpp", None),
            ][len(distractors)]
            if fallback not in distractors:
                distractors.append(fallback)

        candidate_specs: list[tuple[str, EvalTask | None, bool]] = []
        candidate_specs.append((distractors[0][0], distractors[0][1], False))
        candidate_specs.append((positives[0], task, True))
        candidate_specs.append((distractors[1][0], distractors[1][1], False))
        if len(positives) > 1:
            candidate_specs.append((positives[1], task, True))

        candidates_json: list[dict[str, Any]] = []
        ranked_ids: list[int] = []
        for idx, (path, source_task, relevant) in enumerate(candidate_specs, start=1):
            terms = _demo_terms(source_task)[:4] if source_task is not None else []
            preview_parts = [
                f"task={source_task.task_type.value}" for _ in [0] if source_task is not None
            ]
            if terms:
                preview_parts.append("terms=" + ", ".join(terms))
            if not preview_parts:
                preview_parts.append("non-code distractor or unrelated file")
            candidates_json.append(
                {
                    "id": idx,
                    "name": Path(path).name,
                    "source": path,
                    "preview": " | ".join(preview_parts),
                }
            )
            if relevant:
                ranked_ids.append(idx)

        demos.append(
            {
                "query": task.description,
                "max_ranked_ids": len(candidate_specs),
                "candidates_json": json.dumps(candidates_json, ensure_ascii=True),
                "ranked_ids": ranked_ids,
            }
        )
    return demos


def _build_optimizer_rerank_examples(tasks: list[EvalTask], limit: int) -> list[Any]:
    if limit <= 0:
        return []

    try:
        import dspy  # type: ignore[import-not-found]
    except Exception:
        return []

    examples: list[Any] = []
    fallback_noise = [
        "docs/architecture.md",
        "results/benchmark_summary.json",
        "tests/unit/example_test.cpp",
    ]
    for task in tasks[: max(0, int(limit))]:
        positives = _collect_files(task)[:2]
        if not positives:
            continue

        candidate_specs: list[tuple[str, bool, str]] = []
        terms = ", ".join(_demo_terms(task)[:4]) or task.task_type.value
        for path in positives:
            candidate_specs.append((path, True, f"task={task.task_type.value} | terms={terms}"))
        for noise_path in fallback_noise:
            if noise_path not in [p for p, _, _ in candidate_specs]:
                candidate_specs.append((noise_path, False, "noise file unrelated to the query"))
            if len(candidate_specs) >= 5:
                break

        candidates_json: list[dict[str, Any]] = []
        ranked_ids: list[int] = []
        for idx, (path, relevant, preview) in enumerate(candidate_specs, start=1):
            candidates_json.append(
                {
                    "id": idx,
                    "name": Path(path).name,
                    "source": path,
                    "preview": preview,
                }
            )
            if relevant:
                ranked_ids.append(idx)

        examples.append(
            dspy.Example(
                query=task.description,
                max_ranked_ids=min(5, len(candidate_specs)),
                candidates_json=json.dumps(candidates_json, ensure_ascii=True),
                ranked_ids=ranked_ids,
            ).with_inputs("query", "max_ranked_ids", "candidates_json")
        )
    return examples


def _dspy_rerank_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    gold = QueryPlanner._coerce_ranked_ids(getattr(example, "ranked_ids", []) or [], 128)
    pred = QueryPlanner._coerce_ranked_ids(getattr(prediction, "ranked_ids", []) or [], 128)
    if not gold or not pred:
        return 0.0

    gold_set = set(gold)
    top_pred = pred[: len(gold)]
    overlap = sum(1 for item in top_pred if item in gold_set) / max(1, len(gold_set))
    reciprocal = 0.0
    for idx, item in enumerate(pred, start=1):
        if item in gold_set:
            reciprocal = 1.0 / idx
            break
    precision = sum(1 for item in pred if item in gold_set) / max(1, len(pred))
    return max(0.0, min(1.0, 0.5 * overlap + 0.3 * reciprocal + 0.2 * precision))


def _count_predictor_demos(program: Any) -> int:
    try:
        predictors = list(program.named_predictors())
    except Exception:
        return 0

    total = 0
    for _, predictor in predictors:
        demos = getattr(predictor, "demos", None) or []
        total += len(demos)
    return total


def _build_compiled_dspy_reranker(
    config: PipelineConfig, train_tasks: list[EvalTask]
) -> Any | None:
    if not config.use_dspy_retrieval_rerank or not config.dspy_retrieval_optimize:
        return None

    try:
        import dspy  # type: ignore[import-not-found]
    except Exception:
        return None

    model_cfg = config.dspy_retrieval_model or config.critic_model or config.executor_model
    candidates = [model_cfg.name]
    if not model_cfg.name.startswith("openai/"):
        candidates.append(f"openai/{model_cfg.name}")

    lm = None
    for model_name in candidates:
        try:
            lm = dspy.LM(
                model_name,
                api_base=model_cfg.base_url,
                api_key=model_cfg.api_key,
                temperature=0.0,
                max_tokens=max(128, int(config.dspy_retrieval_max_tokens or 16384)),
                timeout=float(model_cfg.request_timeout_s),
            )
            break
        except Exception:
            continue
    if lm is None:
        return None

    trainset = _build_optimizer_rerank_examples(
        train_tasks,
        limit=int(config.dspy_retrieval_optimizer_trainset_size),
    )
    if not trainset:
        return None

    sig = QueryPlanner._build_dspy_signature()
    adapter_candidates: list[Any] = []
    if bool(config.dspy_retrieval_prefer_json) and hasattr(dspy, "JSONAdapter"):
        adapter_candidates.append(dspy.JSONAdapter())
    if hasattr(dspy, "ChatAdapter"):
        adapter_candidates.append(dspy.ChatAdapter())

    for adapter in adapter_candidates:
        optimizer = dspy.BootstrapFewShot(
            metric=_dspy_rerank_metric,
            metric_threshold=float(config.dspy_retrieval_metric_threshold),
            max_bootstrapped_demos=int(config.dspy_retrieval_bootstrapped_demos),
            max_labeled_demos=int(config.dspy_retrieval_labeled_demos),
            max_rounds=1,
        )
        try:
            with dspy.context(lm=lm, adapter=adapter):
                student = dspy.Predict(sig)
                compiled = optimizer.compile(student, trainset=trainset)
            if _count_predictor_demos(compiled) > 0:
                return compiled
        except Exception:
            continue
    return None


def _task_metrics(task: EvalTask, query_results: list[Any]) -> dict[str, float]:
    scored = _ranked_chunks(query_results)
    sources = _unique_sources(scored)
    gold_files = _collect_files(task)
    gold_symbols = _collect_symbols(task)

    metrics: dict[str, float] = {}

    # File-level ranking metrics.
    for k in (1, 3, 5, 10):
        top_sources = sources[:k]
        hit = 0.0
        if gold_files:
            if any(any(_match_file(s, gf) for s in top_sources) for gf in gold_files):
                hit = 1.0
        metrics[f"file_hit_at_{k}"] = hit

        if gold_files:
            matched = {gf for gf in gold_files if any(_match_file(s, gf) for s in top_sources)}
            metrics[f"file_recall_at_{k}"] = len(matched) / max(1, len(gold_files))
        else:
            metrics[f"file_recall_at_{k}"] = 0.0

    mrr = 0.0
    if gold_files:
        for i, s in enumerate(sources, start=1):
            if any(_match_file(s, gf) for gf in gold_files):
                mrr = 1.0 / i
                break
    metrics["file_mrr"] = mrr

    # Symbol hit@k over ranked chunks.
    for k in (1, 3, 5, 10):
        top_chunks = scored[:k]
        text = "\n".join((c.content or "") + "\n" + (c.source or "") for _, _, c in top_chunks)
        if gold_symbols:
            ok = any(_pattern_match(text, p) for p in gold_symbols)
            metrics[f"symbol_hit_at_{k}"] = 1.0 if ok else 0.0
        else:
            metrics[f"symbol_hit_at_{k}"] = 0.0

    # Grep noise rate.
    grep_chunks: list[YAMSChunk] = []
    for r in query_results:
        if r.spec.query_type == QueryType.GREP:
            grep_chunks.extend(list(r.chunks or []))
    noise = 0
    for c in grep_chunks:
        src = c.source or ""
        noisy = _is_noise_source(src)
        if noisy and not any(_match_file(src, gf) for gf in gold_files):
            noise += 1
    metrics["grep_noise_rate"] = (noise / max(1, len(grep_chunks))) if grep_chunks else 0.0

    # Graph expansion utility: useful neighbor ratio.
    graph_chunks: list[YAMSChunk] = []
    for r in query_results:
        if r.spec.query_type == QueryType.GRAPH:
            graph_chunks.extend(list(r.chunks or []))
    useful = 0
    for c in graph_chunks:
        txt = (c.content or "") + "\n" + (c.source or "")
        has_file = any(_match_file(c.source or "", gf) for gf in gold_files)
        has_symbol = any(_pattern_match(txt, p) for p in gold_symbols)
        if has_file or has_symbol:
            useful += 1
    metrics["graph_useful_neighbor_ratio"] = (
        useful / max(1, len(graph_chunks)) if graph_chunks else 0.0
    )

    metrics["query_count"] = float(len(query_results))
    metrics["chunk_count"] = float(sum(len(r.chunks or []) for r in query_results))
    metrics["source_count"] = float(len(sources))
    return metrics


async def _run_task(
    task: EvalTask,
    *,
    decomposer: TaskDecomposer | None,
    planner: QueryPlanner,
    max_queries: int,
    decompose_mode: str,
    use_task_seeding: bool,
):
    t0 = time.perf_counter()
    mode = str(decompose_mode or "heuristic").lower()
    if mode == "model":
        if decomposer is None:
            raise RuntimeError("model decomposition requested but no decomposer is available")
        specs = await decomposer.decompose(
            task.description,
            max_queries=max_queries,
            use_task_seeding=bool(use_task_seeding),
        )
    else:
        specs = _heuristic_decompose(task.description, max_queries=max_queries)
    results = await planner.execute(specs, allow_adaptive=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    metrics = _task_metrics(task, results)
    metrics["retrieval_latency_ms"] = float(dt_ms)
    return {
        "task_id": task.id,
        "task_type": task.task_type.value,
        "spec_count": len(specs),
        "metrics": metrics,
    }


async def _run_suite(
    tasks: list[EvalTask],
    config: PipelineConfig,
    max_queries: int,
    *,
    decompose_mode: str,
    use_task_seeding: bool,
    decomposer_temperature: float,
):
    yams_kwargs = {
        "cwd": config.yams_cwd,
        "search_weights": config.search_weights,
    }
    async with YAMSClient(**yams_kwargs) as yams:
        decomposer: TaskDecomposer | None = None
        if str(decompose_mode or "heuristic").lower() == "model":
            decomposer_cfg = replace(config.executor_model)
            decomposer_cfg.temperature = float(decomposer_temperature)
            decomposer = TaskDecomposer(decomposer_cfg)
        dspy_rerank_model = None
        if config.use_dspy_retrieval_rerank:
            try:
                import dspy  # type: ignore[import-not-found]

                model_cfg = (
                    config.dspy_retrieval_model or config.critic_model or config.executor_model
                )
                candidates = [model_cfg.name]
                if not model_cfg.name.startswith("openai/"):
                    candidates.append(f"openai/{model_cfg.name}")
                for model_name in candidates:
                    try:
                        dspy_rerank_model = dspy.LM(
                            model_name,
                            api_base=model_cfg.base_url,
                            api_key=model_cfg.api_key,
                            temperature=0.0,
                            max_tokens=max(128, int(config.dspy_retrieval_max_tokens or 16384)),
                            timeout=float(model_cfg.request_timeout_s),
                        )
                        break
                    except Exception:
                        continue
            except Exception:
                dspy_rerank_model = None
        planner = QueryPlanner(
            yams,
            dspy_rerank_model=dspy_rerank_model,
            dspy_rerank_top_k=int(config.dspy_retrieval_top_k),
            dspy_rerank_prefer_json=bool(config.dspy_retrieval_prefer_json),
        )
        out = []
        for t in tasks:
            dspy_rerank_predictor = None
            if dspy_rerank_model is not None and config.dspy_retrieval_optimize:
                train_tasks = [task for task in tasks if task.id != t.id]
                dspy_rerank_predictor = _build_compiled_dspy_reranker(config, train_tasks)
            planner = QueryPlanner(
                yams,
                dspy_rerank_model=dspy_rerank_model,
                dspy_rerank_predictor=dspy_rerank_predictor,
                dspy_rerank_top_k=int(config.dspy_retrieval_top_k),
                dspy_rerank_demos=_build_dspy_rerank_demos(
                    tasks,
                    current_task=t,
                    limit=int(config.dspy_retrieval_demo_count),
                ),
                dspy_rerank_prefer_json=bool(config.dspy_retrieval_prefer_json),
            )
            out.append(
                await _run_task(
                    t,
                    decomposer=decomposer,
                    planner=planner,
                    max_queries=max_queries,
                    decompose_mode=decompose_mode,
                    use_task_seeding=use_task_seeding,
                )
            )
        return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "file_hit_at_1",
        "file_hit_at_3",
        "file_hit_at_5",
        "file_recall_at_5",
        "file_mrr",
        "symbol_hit_at_5",
        "grep_noise_rate",
        "graph_useful_neighbor_ratio",
        "retrieval_latency_ms",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [float((r.get("metrics") or {}).get(k, 0.0)) for r in rows]
        out[k] = statistics.fmean(vals) if vals else 0.0
    return out


def main() -> int:
    default_task_dir, default_models_cfg, default_yams_cwd = _default_paths()

    parser = argparse.ArgumentParser(description="Run retrieval-only benchmark (no generation)")
    parser.add_argument("--task-dir", default=default_task_dir)
    parser.add_argument("--task-type", default=None, help="qa|coding")
    parser.add_argument("--tags", default="", help="Comma-separated tag filter")
    parser.add_argument(
        "--tag-match",
        choices=["all", "any"],
        default="all",
        help="Tag matching mode (all tags required or any tag)",
    )
    parser.add_argument("--models-config", default=default_models_cfg)
    parser.add_argument("--executor", default=None, help="Executor model key for decomposer")
    parser.add_argument("--critic", default=None, help="Critic model key (recorded only)")
    parser.add_argument("--yams-cwd", default=default_yams_cwd)
    parser.add_argument("--max-queries", type=int, default=6)
    parser.add_argument(
        "--decompose-mode",
        choices=["heuristic", "model"],
        default="heuristic",
        help="Use deterministic heuristic decomposition or model-based decomposition",
    )
    parser.add_argument(
        "--task-seeding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable task-specific hardcoded query seeds during decomposition",
    )
    parser.add_argument(
        "--decomposer-temperature",
        type=float,
        default=0.0,
        help="Temperature for model decomposition mode",
    )
    parser.add_argument(
        "--prime-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-index repo content and wait for ingestion to settle before retrieval benchmark",
    )
    parser.add_argument(
        "--prime-timeout-s",
        type=float,
        default=900.0,
        help="Timeout for pre-benchmark indexing wait",
    )
    parser.add_argument(
        "--dspy-retrieval-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable DSPy reranking over top-K retrieval candidates",
    )
    parser.add_argument(
        "--dspy-retrieval-top-k",
        type=int,
        default=5,
        help="Top-K candidates to rerank with DSPy when enabled",
    )
    parser.add_argument(
        "--dspy-retrieval-demo-count",
        type=int,
        default=0,
        help="Number of cross-task DSPy reranker demos to include (experimental; default off)",
    )
    parser.add_argument(
        "--dspy-retrieval-optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the DSPy reranker with BootstrapFewShot before benchmarking",
    )
    parser.add_argument(
        "--dspy-retrieval-trainset-size",
        type=int,
        default=6,
        help="Number of eval tasks to convert into DSPy optimizer examples",
    )
    parser.add_argument(
        "--dspy-retrieval-adapter",
        choices=["json", "chat"],
        default="json",
        help="Preferred DSPy adapter for reranking",
    )
    parser.add_argument(
        "--dspy-retrieval-max-tokens",
        type=int,
        default=16384,
        help="DSPy reranker max output tokens",
    )
    parser.add_argument(
        "--dspy-retrieval-model",
        default="openai/gpt-oss-20b",
        help="DSPy reranker model key or raw model id",
    )
    parser.add_argument("--out", default=None)
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
    exec_key = args.executor or defaults.get("executor", "gpt-oss-20b")
    critic_key = args.critic or defaults.get("critic", "qwen35-35b-a3b")

    config = PipelineConfig(
        executor_model=_build_model_config_or_id(
            models_cfg, exec_key, preferred_role="executor", default_temperature=1.0
        ),
        critic_model=_build_model_config_or_id(
            models_cfg, critic_key, preferred_role="critic", default_temperature=0.0
        ),
        yams_cwd=args.yams_cwd,
        use_dspy_retrieval_rerank=bool(args.dspy_retrieval_rerank),
        dspy_retrieval_top_k=int(args.dspy_retrieval_top_k),
        dspy_retrieval_max_tokens=int(args.dspy_retrieval_max_tokens),
        dspy_retrieval_demo_count=int(args.dspy_retrieval_demo_count),
        dspy_retrieval_prefer_json=str(args.dspy_retrieval_adapter) == "json",
        dspy_retrieval_optimize=bool(args.dspy_retrieval_optimize),
        dspy_retrieval_optimizer_trainset_size=int(args.dspy_retrieval_trainset_size),
        dspy_retrieval_model=_build_model_config_or_id(
            models_cfg,
            str(args.dspy_retrieval_model),
            preferred_role="critic",
            default_temperature=0.0,
        ),
    )

    runner = EvalRunner(PipelineConfig(), task_dir=args.task_dir)
    tasks = runner.load_tasks(args.task_dir, task_type=_task_type_from_str(args.task_type))
    tags = {t.strip().lower() for t in args.tags.split(",") if t.strip()}
    tasks = _filter_by_tags(tasks, tags, mode=str(args.tag_match))
    if not tasks:
        console.print("No tasks found for selection")
        return 1

    rows = asyncio.run(
        _run_suite(
            tasks,
            config,
            max_queries=int(args.max_queries),
            decompose_mode=str(args.decompose_mode),
            use_task_seeding=bool(args.task_seeding),
            decomposer_temperature=float(args.decomposer_temperature),
        )
    )
    summary = _summary(rows)

    tbl = Table(title="Retrieval Benchmark")
    tbl.add_column("Task")
    tbl.add_column("File@3")
    tbl.add_column("Recall@5")
    tbl.add_column("MRR")
    tbl.add_column("Symbol@5")
    tbl.add_column("GrepNoise")
    tbl.add_column("GraphUse")
    tbl.add_column("Latency ms")
    for r in rows:
        m = r["metrics"]
        tbl.add_row(
            r["task_id"],
            f"{m['file_hit_at_3']:.2f}",
            f"{m['file_recall_at_5']:.2f}",
            f"{m['file_mrr']:.2f}",
            f"{m['symbol_hit_at_5']:.2f}",
            f"{m['grep_noise_rate']:.2f}",
            f"{m['graph_useful_neighbor_ratio']:.2f}",
            f"{m['retrieval_latency_ms']:.0f}",
        )
    console.print(tbl)

    console.print(
        "\nSummary: "
        f"File@3={summary['file_hit_at_3']:.2f} "
        f"Recall@5={summary['file_recall_at_5']:.2f} "
        f"MRR={summary['file_mrr']:.2f} "
        f"Symbol@5={summary['symbol_hit_at_5']:.2f} "
        f"GrepNoise={summary['grep_noise_rate']:.2f} "
        f"GraphUse={summary['graph_useful_neighbor_ratio']:.2f}"
    )

    payload = {
        "config": {
            "executor": exec_key,
            "critic": critic_key,
            "executor_model_name": config.executor_model.name,
            "critic_model_name": config.critic_model.name if config.critic_model else None,
            "task_dir": args.task_dir,
            "task_type": args.task_type,
            "tags": sorted(tags),
            "max_queries": int(args.max_queries),
            "yams_cwd": args.yams_cwd,
            "decompose_mode": str(args.decompose_mode),
            "task_seeding": bool(args.task_seeding),
            "decomposer_temperature": float(args.decomposer_temperature),
            "prime_index": bool(args.prime_index),
            "dspy_retrieval_rerank": bool(args.dspy_retrieval_rerank),
            "dspy_retrieval_top_k": int(args.dspy_retrieval_top_k),
            "dspy_retrieval_max_tokens": int(args.dspy_retrieval_max_tokens),
            "dspy_retrieval_demo_count": int(args.dspy_retrieval_demo_count),
            "dspy_retrieval_optimize": bool(args.dspy_retrieval_optimize),
            "dspy_retrieval_trainset_size": int(args.dspy_retrieval_trainset_size),
            "dspy_retrieval_adapter": str(args.dspy_retrieval_adapter),
            "dspy_retrieval_model": str(args.dspy_retrieval_model),
            "dspy_retrieval_model_name": (
                config.dspy_retrieval_model.name if config.dspy_retrieval_model else None
            ),
            "tag_match": str(args.tag_match),
        },
        "summary": summary,
        "tasks": rows,
    }
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote retrieval benchmark: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
