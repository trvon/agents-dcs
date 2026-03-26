from __future__ import annotations

from dcs.indexing import _status_is_retrieval_ready


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
