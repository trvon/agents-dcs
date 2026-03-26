from __future__ import annotations

from pathlib import Path

from dcs.indexing import (
    _index_fingerprint,
    _state_matches_fingerprint,
    _status_is_retrieval_ready,
    _sync_add_command,
)


def test_status_is_retrieval_ready_when_core_services_ready_and_queue_empty() -> None:
    status = {
        "ready": True,
        "readiness": {
            "content_store": True,
            "database": True,
            "metadata_repo": True,
            "search_engine": True,
        },
        "post_ingest": {
            "queued": 0,
            "inflight": 0,
            "deferred_queue_depth": 0,
            "rpc": {"queued": 0},
        },
    }
    assert _status_is_retrieval_ready(status)


def test_status_is_not_ready_when_post_ingest_queue_nonzero() -> None:
    status = {
        "ready": True,
        "readiness": {
            "content_store": True,
            "database": True,
            "metadata_repo": True,
            "search_engine": True,
        },
        "post_ingest": {
            "queued": 1,
            "inflight": 0,
            "deferred_queue_depth": 0,
            "rpc": {"queued": 0},
        },
    }
    assert not _status_is_retrieval_ready(status)


def test_index_fingerprint_changes_when_tracked_file_changes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "demo.py"
    target.write_text("print('a')\n", encoding="utf-8")

    fp1, count1 = _index_fingerprint(tmp_path, include="*.py", exclude="")
    target.write_text("print('b')\n", encoding="utf-8")
    fp2, count2 = _index_fingerprint(tmp_path, include="*.py", exclude="")

    assert count1 == 1
    assert count2 == 1
    assert fp1 != fp2


def test_index_fingerprint_ignores_excluded_files(tmp_path: Path) -> None:
    keep = tmp_path / "keep.py"
    skip_dir = tmp_path / "build"
    skip_dir.mkdir()
    skip = skip_dir / "skip.py"
    keep.write_text("print('keep')\n", encoding="utf-8")
    skip.write_text("print('skip')\n", encoding="utf-8")

    fp1, count1 = _index_fingerprint(tmp_path, include="*.py", exclude="build/**")
    skip.write_text("print('changed')\n", encoding="utf-8")
    fp2, count2 = _index_fingerprint(tmp_path, include="*.py", exclude="build/**")

    assert count1 == 1
    assert count2 == 1
    assert fp1 == fp2


def test_state_matches_fingerprint_requires_same_filters() -> None:
    state = {"fingerprint": "abc", "include": "*.py", "exclude": "build/**"}
    assert _state_matches_fingerprint(state, fingerprint="abc", include="*.py", exclude="build/**")
    assert not _state_matches_fingerprint(
        state,
        fingerprint="def",
        include="*.py",
        exclude="build/**",
    )
    assert not _state_matches_fingerprint(
        state,
        fingerprint="abc",
        include="*.cpp",
        exclude="build/**",
    )


def test_sync_add_command_uses_sync_flags() -> None:
    cmd = _sync_add_command(
        yams_binary="yams",
        include="*.py",
        exclude="build/**",
        sync_timeout_s=120,
    )
    assert "--sync" in cmd
    assert "--sync-timeout" in cmd
    assert "120" in cmd
