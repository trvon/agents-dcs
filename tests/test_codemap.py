from __future__ import annotations

import pytest

from dcs.codemap import CodemapBuilder
from dcs.types import YAMSChunk


class _FakeClient:
    async def graph_query(self, **kwargs):
        if kwargs.get("list_types"):
            return {"node_type_counts": {"file": 10, "function": 25, "class": 3}}
        if kwargs.get("list_type") == "file":
            return {
                "connected_nodes": [
                    {
                        "nodeKey": "path:file:/repo/src/foo.cpp",
                        "type": "file",
                        "label": "/repo/src/foo.cpp",
                    },
                    {
                        "nodeKey": "path:file:/repo/include/foo.h",
                        "type": "file",
                        "label": "/repo/include/foo.h",
                    },
                ]
            }
        if kwargs.get("relation") == "contains":
            path = kwargs.get("node_key", "")
            return {
                "connected_nodes": [
                    *(
                        [
                            {
                                "nodeKey": "function:Foo@/repo/src/foo.cpp",
                                "type": "function_version",
                                "label": "ns::Foo@/repo/src/foo.cpp",
                            },
                            {
                                "nodeKey": "class:Bar@/repo/src/foo.cpp",
                                "type": "class_version",
                                "label": "ns::Bar@/repo/src/foo.cpp",
                            },
                        ]
                        if str(path).endswith("foo.cpp")
                        else [
                            {
                                "nodeKey": "function:FooHeader@/repo/include/foo.h",
                                "type": "function_version",
                                "label": "ns::FooHeader@/repo/include/foo.h",
                            }
                        ]
                    ),
                ],
                "total_edges_traversed": 2 if str(path).endswith("foo.cpp") else 1,
            }
        return {}

    async def search(self, query, limit=10, **kwargs):  # pragma: no cover
        return []


@pytest.mark.asyncio
async def test_codemap_builder_accepts_snake_case_graph_fields() -> None:
    builder = CodemapBuilder(
        _FakeClient(),
        token_budget=512,
        max_files=5,
        max_symbols_per_file=5,
        include_type_counts=True,
    )
    result = await builder.build(task="describe foo")
    assert result.node_type_counts["file"] == 10
    assert result.node_count >= 2
    assert "/repo/src/" in result.tree_text
    assert "[fn] Foo" in result.tree_text


class _FallbackClient(_FakeClient):
    async def graph_query(self, **kwargs):
        if kwargs.get("list_types"):
            return {"node_type_counts": {"file": 0}}
        if kwargs.get("list_type") == "file":
            return {"connected_nodes": []}
        if kwargs.get("relation") == "contains":
            return {"connected_nodes": []}
        return {}

    async def search(self, query, limit=10, **kwargs):
        return [
            YAMSChunk(chunk_id="1", content="x", source="/repo/src/good.cpp", score=0.9),
            YAMSChunk(
                chunk_id="2", content="x", source="/repo/external/agent/results/bad.json", score=0.8
            ),
            YAMSChunk(chunk_id="3", content="x", source="/repo/include/good.h", score=0.7),
        ]


@pytest.mark.asyncio
async def test_codemap_search_fallback_filters_non_code_noise() -> None:
    builder = CodemapBuilder(
        _FallbackClient(), token_budget=512, max_files=5, max_symbols_per_file=5
    )
    nodes = await builder._get_file_nodes(None, "describe foo")
    labels = [n.label for n in nodes]
    assert "/repo/src/good.cpp" in labels
    assert "/repo/include/good.h" in labels
    assert all("external/agent/results" not in label for label in labels)


@pytest.mark.asyncio
async def test_codemap_prefers_search_anchored_focus_paths() -> None:
    builder = CodemapBuilder(
        _FallbackClient(), token_budget=512, max_files=5, max_symbols_per_file=5
    )
    paths = await builder._select_focus_paths("describe foo")
    assert paths == ["/repo/src/good.cpp", "/repo/include/good.h"]


def test_codemap_task_term_matching_prefers_relevant_paths() -> None:
    builder = CodemapBuilder(_FallbackClient(), token_budget=512)
    assert builder._path_matches_task(
        "/repo/src/vector/embedding_service.cpp",
        "How does the EmbeddingService work?",
    )
    assert not builder._path_matches_task(
        "/repo/src/crypto/sha256_hasher.cpp",
        "How does the EmbeddingService work?",
    )
