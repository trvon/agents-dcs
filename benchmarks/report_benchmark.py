from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def _load_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid benchmark payload: {path}")
    return raw


def _iter_model_rows(payload: dict[str, Any], source_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if "tasks" in payload and isinstance(payload.get("tasks"), list):
        config = payload.get("config") or {}
        model = str(config.get("executor") or config.get("model") or Path(source_path).stem)
        for task in payload["tasks"]:
            if isinstance(task, dict):
                rows.append({"model": model, "source_path": source_path, **task})
        return rows

    for model, tasks in payload.items():
        if not isinstance(model, str) or not isinstance(tasks, list):
            continue
        for task in tasks:
            if isinstance(task, dict):
                rows.append({"model": model, "source_path": source_path, **task})
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarize_models(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model") or "unknown"), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for model, items in grouped.items():
        pass_vals = [1.0 if bool(item.get("passed", False)) else 0.0 for item in items]
        quality_vals = [
            float((item.get("metrics") or {}).get("quality_score", 0.0)) for item in items
        ]
        latency_vals = [
            float((item.get("metrics") or {}).get("total_latency_ms", 0.0)) for item in items
        ]
        faith_vals = [
            float((item.get("metrics") or {}).get("faithfulness_confidence", 0.0)) for item in items
        ]
        plan_vals = [
            float((item.get("metrics") or {}).get("plan_coverage", 0.0))
            for item in items
            if "plan_coverage" in (item.get("metrics") or {})
        ]
        out[model] = {
            "task_count": len(items),
            "pass_rate_mean": _mean(pass_vals),
            "pass_rate_stdev": _stdev(pass_vals),
            "quality_mean": _mean(quality_vals),
            "quality_stdev": _stdev(quality_vals),
            "latency_mean": _mean(latency_vals),
            "latency_stdev": _stdev(latency_vals),
            "faithfulness_mean": _mean(faith_vals),
            "plan_coverage_mean": _mean(plan_vals),
        }
    return out


def compare_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model") or "unknown")
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        grouped.setdefault(task_id, {})[model] = row

    models = sorted({str(row.get("model") or "unknown") for row in rows})
    if len(models) < 2:
        return {"wins": {}, "tasks_compared": 0}

    wins = {model: 0 for model in models}
    ties = 0
    compared = 0
    for task_id, entries in grouped.items():
        if len(entries) < 2:
            continue
        compared += 1
        ranked = sorted(
            entries.items(),
            key=lambda item: (
                1.0 if bool(item[1].get("passed", False)) else 0.0,
                float((item[1].get("metrics") or {}).get("quality_score", 0.0)),
                -float((item[1].get("metrics") or {}).get("total_latency_ms", 0.0)),
            ),
            reverse=True,
        )
        if len(ranked) >= 2:
            top_score = (
                1.0 if bool(ranked[0][1].get("passed", False)) else 0.0,
                float((ranked[0][1].get("metrics") or {}).get("quality_score", 0.0)),
                -float((ranked[0][1].get("metrics") or {}).get("total_latency_ms", 0.0)),
            )
            second_score = (
                1.0 if bool(ranked[1][1].get("passed", False)) else 0.0,
                float((ranked[1][1].get("metrics") or {}).get("quality_score", 0.0)),
                -float((ranked[1][1].get("metrics") or {}).get("total_latency_ms", 0.0)),
            )
            if top_score == second_score:
                ties += 1
            else:
                wins[ranked[0][0]] += 1
    return {"wins": wins, "ties": ties, "tasks_compared": compared}


def _render(
    console: Console, summary: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> None:
    tbl = Table(title="Benchmark Report")
    tbl.add_column("Model")
    tbl.add_column("Tasks", justify="right")
    tbl.add_column("PassMean", justify="right")
    tbl.add_column("QualityMean", justify="right")
    tbl.add_column("LatencyMean", justify="right")
    tbl.add_column("FaithMean", justify="right")
    tbl.add_column("PlanMean", justify="right")
    for model, stats in sorted(summary.items()):
        tbl.add_row(
            model,
            str(int(stats.get("task_count", 0))),
            f"{float(stats.get('pass_rate_mean', 0.0)):.2f}",
            f"{float(stats.get('quality_mean', 0.0)):.2f}",
            f"{float(stats.get('latency_mean', 0.0)):.0f}",
            f"{float(stats.get('faithfulness_mean', 0.0)):.2f}",
            f"{float(stats.get('plan_coverage_mean', 0.0)):.2f}",
        )
    console.print(tbl)

    win_tbl = Table(title="Paired Comparison")
    win_tbl.add_column("Model")
    win_tbl.add_column("Wins", justify="right")
    for model, wins in sorted((comparison.get("wins") or {}).items()):
        win_tbl.add_row(model, str(int(wins)))
    win_tbl.add_row("ties", str(int(comparison.get("ties", 0))))
    win_tbl.add_row("tasks_compared", str(int(comparison.get("tasks_compared", 0))))
    console.print(win_tbl)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sequential benchmark runs into one report")
    parser.add_argument("inputs", nargs="+", help="Benchmark JSON files")
    parser.add_argument("--out", default=None, help="Write merged report JSON")
    args = parser.parse_args()

    console = Console()
    rows: list[dict[str, Any]] = []
    for fp in args.inputs:
        path = Path(fp)
        payload = _load_payload(path)
        rows.extend(_iter_model_rows(payload, str(path)))

    summary = summarize_models(rows)
    comparison = compare_models(rows)
    _render(console, summary, comparison)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "summary": summary,
                    "comparison": comparison,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        console.print(f"Wrote benchmark report: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
