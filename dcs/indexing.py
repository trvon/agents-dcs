from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_INDEX_INCLUDE = "*.cpp,*.hpp,*.h,*.cc,*.cxx,*.c,*.py,*.md,*.yaml,*.yml,*.toml"
DEFAULT_INDEX_EXCLUDE = ".git/**,build/**,dist/**,.venv/**,venv/**,__pycache__/**,node_modules/**"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _status_is_retrieval_ready(status: dict[str, Any], *, max_pending: int = 0) -> bool:
    readiness = status.get("readiness") if isinstance(status, dict) else {}
    if not isinstance(readiness, dict):
        readiness = {}

    post_ingest = status.get("post_ingest") if isinstance(status, dict) else {}
    if not isinstance(post_ingest, dict):
        post_ingest = {}

    rpc = post_ingest.get("rpc") if isinstance(post_ingest.get("rpc"), dict) else {}
    pending = (
        _as_int(post_ingest.get("queued"))
        + _as_int(post_ingest.get("inflight"))
        + _as_int(post_ingest.get("deferred_queue_depth"))
        + _as_int(rpc.get("queued"))
    )

    return (
        bool(status.get("ready", False))
        and bool(readiness.get("content_store", True))
        and bool(readiness.get("database", True))
        and bool(readiness.get("metadata_repo", True))
        and bool(readiness.get("search_engine", True))
        and pending <= int(max_pending)
    )


def _run_command(
    command: list[str],
    *,
    cwd: str,
    timeout_s: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_s)),
        check=False,
    )
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit {proc.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)} :: {detail}")
    return proc


def _status_json(yams_binary: str, *, cwd: str, timeout_s: float) -> dict[str, Any]:
    proc = _run_command([yams_binary, "status", "--json"], cwd=cwd, timeout_s=timeout_s)
    text = (proc.stdout or "").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        return {}
    return data


def prime_yams_index(
    *,
    root: str,
    yams_binary: str = "yams",
    include: str = DEFAULT_INDEX_INCLUDE,
    exclude: str = DEFAULT_INDEX_EXCLUDE,
    timeout_s: float = 900.0,
    poll_interval_s: float = 1.0,
    max_pending: int = 0,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Index root not found: {root_path}")

    _run_command(
        [
            yams_binary,
            "add",
            ".",
            "--recursive",
            f"--include={include}",
            f"--exclude={exclude}",
            "--tags",
            "code,benchmark",
        ],
        cwd=str(root_path),
        timeout_s=min(max(30.0, float(timeout_s)), 300.0),
    )

    deadline = time.monotonic() + max(5.0, float(timeout_s))
    stable_polls = 0
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_status = _status_json(yams_binary, cwd=str(root_path), timeout_s=30.0)
        if _status_is_retrieval_ready(last_status, max_pending=max_pending):
            stable_polls += 1
            if stable_polls >= 2:
                return last_status
        else:
            stable_polls = 0
        time.sleep(max(0.2, float(poll_interval_s)))

    raise TimeoutError(f"Timed out waiting for YAMS ingest to settle under {root_path}")


__all__ = [
    "DEFAULT_INDEX_EXCLUDE",
    "DEFAULT_INDEX_INCLUDE",
    "prime_yams_index",
    "_status_is_retrieval_ready",
]
