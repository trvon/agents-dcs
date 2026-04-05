from __future__ import annotations

from pathlib import Path

from dcs.client import YAMSClient
from dcs.types import YAMSChunk


def test_split_grep_pattern_supports_include_exclude_tokens() -> None:
    pat, path_hint, includes, excludes = YAMSClient._split_grep_pattern(
        "registerTool path:src/mcp/mcp_server.cpp include:src/mcp/* exclude:tests/*"
    )
    assert pat == "registerTool"
    assert path_hint == "src/mcp/mcp_server.cpp"
    assert includes == ["src/mcp/*"]
    assert excludes == ["tests/*"]


def test_source_matches_filters_honors_hints() -> None:
    src = "/Users/trevon/work/tools/yams/src/mcp/mcp_server.cpp"
    assert YAMSClient._source_matches_filters(
        src,
        cwd="/Users/trevon/work/tools/yams",
        include_hints=["*src/mcp/*"],
        exclude_hints=["*tests/*"],
    )
    assert not YAMSClient._source_matches_filters(
        "/Users/trevon/work/tools/yams/tests/unit/foo_test.cpp",
        cwd="/Users/trevon/work/tools/yams",
        include_hints=["*src/*"],
    )


def test_parse_graph_query_builds_node_key_for_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "src" / "mcp" / "mcp_server.cpp"
    target.parent.mkdir(parents=True)
    target.write_text("// test\n", encoding="utf-8")

    c = YAMSClient(cwd=str(root))
    args = c._parse_graph_query("src/mcp/mcp_server.cpp depth:2 limit:33")
    assert args["depth"] == 2
    assert args["limit"] == 33
    assert "node_key" in args
    assert str(target.resolve()) in str(args["node_key"])


def test_chunks_from_search_data_uses_snippet_path_when_result_path_is_artifact() -> None:
    data = {
        "results": [
            {
                "id": "1",
                "path": "/repo/external/agent/results/checkpoint.json",
                "snippet": "/repo/src/daemon/components/EmbeddingService.cpp",
                "score": 0.9,
            }
        ]
    }
    chunks = YAMSClient._chunks_from_search_data(data)
    assert len(chunks) == 1
    assert chunks[0].source == "/repo/src/daemon/components/EmbeddingService.cpp"


def test_extract_tool_data_accepts_structured_content_without_data_wrapper() -> None:
    payload = {
        "structuredContent": {
            "results": [{"path": "/repo/src/foo.cpp", "snippet": "foo", "score": 0.8}]
        }
    }
    extracted = YAMSClient._extract_tool_data(payload)
    assert isinstance(extracted, dict)
    assert "results" in extracted


def test_rerank_code_chunks_prefers_code_authority_and_identifier_overlap() -> None:
    chunks = [
        YAMSChunk(
            chunk_id="1",
            content="EmbeddingService details",
            score=0.4,
            source="/repo/docs/embedding_service.md",
        ),
        YAMSChunk(
            chunk_id="2",
            content="# /repo/src/vector/embedding_service.cpp\nclass EmbeddingService",
            score=0.3,
            source="/repo/src/vector/embedding_service.cpp",
        ),
    ]
    ranked = YAMSClient._rerank_code_chunks("How does the EmbeddingService work?", chunks)
    assert ranked[0].source == "/repo/src/vector/embedding_service.cpp"


def test_rerank_code_chunks_penalizes_results_artifacts() -> None:
    chunks = [
        YAMSChunk(
            chunk_id="1",
            content="mcp_server.cpp result artifact",
            score=0.9,
            source="/repo/external/agent/results/mcp_server.json",
        ),
        YAMSChunk(
            chunk_id="2",
            content="# /repo/src/mcp/mcp_server.cpp\nregisterTool",
            score=0.4,
            source="/repo/src/mcp/mcp_server.cpp",
        ),
    ]
    ranked = YAMSClient._rerank_code_chunks(
        "tool names registered in mcp_server.cpp",
        chunks,
    )
    assert ranked[0].source == "/repo/src/mcp/mcp_server.cpp"


def test_query_terms_split_camel_case_identifiers() -> None:
    terms = YAMSClient._query_terms("How does the EmbeddingService work?")
    assert "embeddingservice" in terms
    assert "embedding" in terms
    assert "service" in terms
