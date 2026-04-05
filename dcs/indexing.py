from __future__ import annotations

import fnmatch
import hashlib
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


def _try_status_json(
    yams_binary: str,
    *,
    cwd: str,
    timeout_s: float,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            return _status_json(yams_binary, cwd=cwd, timeout_s=timeout_s)
        except Exception as e:
            last_error = e
            if attempt >= attempts - 1:
                break
            time.sleep(max(0.2, float(retry_backoff_s)) * (attempt + 1))
    if last_error is not None:
        return {}
    return {}


def _split_patterns(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _iter_indexed_files(root: Path, *, include: str, exclude: str) -> list[Path]:
    include_patterns = _split_patterns(include)
    exclude_patterns = _split_patterns(exclude)
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if include_patterns and not _matches_any(rel, include_patterns):
            continue
        if exclude_patterns and _matches_any(rel, exclude_patterns):
            continue
        files.append(path)
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def _index_fingerprint(root: Path, *, include: str, exclude: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = _iter_indexed_files(root, include=include, exclude=exclude)
    for path in files:
        rel = path.relative_to(root).as_posix()
        st = path.stat()
        digest.update(rel.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(str(int(st.st_size)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(st.st_mtime_ns)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _state_path(root: Path) -> Path:
    git_dir = root / ".git"
    if git_dir.exists() and git_dir.is_dir():
        return git_dir / "dcs_yams_index_state.json"
    return root / ".dcs_yams_index_state.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _state_matches_fingerprint(
    state: dict[str, Any], *, fingerprint: str, include: str, exclude: str
) -> bool:
    return (
        str(state.get("fingerprint") or "") == str(fingerprint)
        and str(state.get("include") or "") == str(include)
        and str(state.get("exclude") or "") == str(exclude)
    )


def _with_prime_metadata(
    status: dict[str, Any],
    *,
    fingerprint: str,
    file_count: int,
    skipped_add: bool,
    waited: bool,
    reason: str,
) -> dict[str, Any]:
    out = dict(status or {})
    out["_dcs_prime"] = {
        "fingerprint": fingerprint,
        "file_count": int(file_count),
        "skipped_add": bool(skipped_add),
        "waited": bool(waited),
        "reason": str(reason),
    }
    return out


def _sync_add_command(
    *,
    yams_binary: str,
    include: str,
    exclude: str,
    sync_timeout_s: float,
) -> list[str]:
    return [
        yams_binary,
        "add",
        ".",
        "--recursive",
        f"--include={include}",
        f"--exclude={exclude}",
        "--tags",
        "code,benchmark",
        "--sync",
        "--sync-timeout",
        str(int(max(30.0, float(sync_timeout_s)))),
        "--daemon-ready-timeout-ms",
        "30000",
    ]


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

    fingerprint, file_count = _index_fingerprint(root_path, include=include, exclude=exclude)
    state_path = _state_path(root_path)
    state = _load_state(state_path)
    unchanged = _state_matches_fingerprint(
        state,
        fingerprint=fingerprint,
        include=str(include),
        exclude=str(exclude),
    )
    state_phase = str(state.get("phase") or "")
    if unchanged:
        current_status = _try_status_json(
            yams_binary, cwd=str(root_path), timeout_s=20.0, retries=0
        )
        if _status_is_retrieval_ready(current_status, max_pending=max_pending):
            return _with_prime_metadata(
                current_status,
                fingerprint=fingerprint,
                file_count=file_count,
                skipped_add=True,
                waited=False,
                reason="unchanged-ready",
            )
        if state_phase == "in_progress":
            # A prior sync add already started; do not re-add and do not poll forever.
            return _with_prime_metadata(
                current_status,
                fingerprint=fingerprint,
                file_count=file_count,
                skipped_add=True,
                waited=False,
                reason="unchanged-in-progress",
            )
        return _with_prime_metadata(
            current_status,
            fingerprint=fingerprint,
            file_count=file_count,
            skipped_add=True,
            waited=False,
            reason="unchanged-assumed-ready",
        )

    _write_state(
        state_path,
        {
            "fingerprint": fingerprint,
            "file_count": int(file_count),
            "include": str(include),
            "exclude": str(exclude),
            "updated_at": time.time(),
            "phase": "in_progress",
        },
    )
    _run_command(
        _sync_add_command(
            yams_binary=yams_binary,
            include=include,
            exclude=exclude,
            sync_timeout_s=min(max(30.0, float(timeout_s)), 300.0),
        ),
        cwd=str(root_path),
        timeout_s=min(max(45.0, float(timeout_s) + 15.0), 330.0),
    )
    waited_status = _try_status_json(yams_binary, cwd=str(root_path), timeout_s=20.0, retries=0)

    _write_state(
        state_path,
        {
            "fingerprint": fingerprint,
            "file_count": int(file_count),
            "include": str(include),
            "exclude": str(exclude),
            "updated_at": time.time(),
            "phase": "ready",
        },
    )
    return _with_prime_metadata(
        waited_status,
        fingerprint=fingerprint,
        file_count=file_count,
        skipped_add=False,
        waited=True,
        reason="changed-reindexed-sync",
    )


__all__ = [
    "DEFAULT_INDEX_EXCLUDE",
    "DEFAULT_INDEX_INCLUDE",
    "prime_yams_index",
    "_status_is_retrieval_ready",
]
