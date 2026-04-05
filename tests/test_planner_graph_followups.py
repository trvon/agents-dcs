from __future__ import annotations

import pytest

from dcs.planner import QueryPlanner
from dcs.types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult


class _StubYAMS:
    async def search(self, query, limit=10, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def grep(self, pattern, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def graph(self, query):  # pragma: no cover - protocol shim
        return []

    async def get(self, name_or_hash):  # pragma: no cover - protocol shim
        return None

    async def list_docs(self, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def execute_spec(self, spec):  # pragma: no cover - not used in this unit test
        return YAMSQueryResult(spec=spec, chunks=[])


def test_adaptive_followups_add_graph_from_file_chunks() -> None:
    planner = QueryPlanner(_StubYAMS())
    res = YAMSQueryResult(
        spec=QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.9),
        chunks=[
            YAMSChunk(
                chunk_id="c1",
                content="registerTool",
                score=0.9,
                source="/Users/trevon/work/tools/yams/src/mcp/mcp_server.cpp",
                metadata={"enriched": True},
            )
        ],
    )

    follow = planner._adaptive_followups([res])
    graph = [s for s in follow if s.query_type == QueryType.GRAPH]
    assert graph
    assert "mcp_server.cpp" in graph[0].query


def test_validated_paths_only_keep_high_confidence_anchors() -> None:
    planner = QueryPlanner(_StubYAMS())
    res = YAMSQueryResult(
        spec=QuerySpec(
            query="registerTool path:src/mcp/mcp_server.cpp",
            query_type=QueryType.GREP,
            importance=0.9,
        ),
        chunks=[
            YAMSChunk(
                chunk_id="good",
                content="registerTool registration",
                score=0.9,
                source="/repo/src/mcp/mcp_server.cpp",
            ),
            YAMSChunk(
                chunk_id="bad",
                content="unrelated docs",
                score=0.2,
                source="/repo/docs/mcp.md",
            ),
        ],
    )

    paths = planner._validated_paths_from_results([res], min_confidence=0.6, per_result=1)
    assert paths == ["/repo/src/mcp/mcp_server.cpp"]


class _StageStubYAMS:
    def __init__(self) -> None:
        self.calls: list[tuple[QueryType, str]] = []

    async def search(self, query, limit=10, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def grep(self, pattern, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def graph(self, query):  # pragma: no cover - protocol shim
        return []

    async def get(self, name_or_hash):  # pragma: no cover - protocol shim
        return None

    async def list_docs(self, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def execute_spec(self, spec):
        self.calls.append((spec.query_type, spec.query))
        if spec.query_type == QueryType.SEMANTIC:
            return YAMSQueryResult(
                spec=spec,
                chunks=[
                    YAMSChunk(
                        chunk_id="s1",
                        content="semantic hit",
                        source="/repo/src/mcp/mcp_server.cpp",
                        score=0.95,
                    )
                ],
            )
        if spec.query_type == QueryType.GRAPH:
            return YAMSQueryResult(
                spec=spec,
                chunks=[
                    YAMSChunk(
                        chunk_id="g1",
                        content="[file] /repo/src/mcp/mcp_server.cpp",
                        source="/repo/src/mcp/mcp_server.cpp",
                        score=0.9,
                    )
                ],
            )
        return YAMSQueryResult(spec=spec, chunks=[])


class _FakeDSPyPredictor:
    last_demos = None

    def __init__(self):
        self.demos = []

    def __call__(self, **kwargs):
        _FakeDSPyPredictor.last_demos = list(self.demos)

        class _Pred:
            ranked_ids = "[2, 1]"
            rationale = "prefer second file"

        return _Pred()


@pytest.mark.asyncio
async def test_execute_runs_search_then_graph_then_grep_get() -> None:
    yams = _StageStubYAMS()
    planner = QueryPlanner(yams)
    specs = [
        QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.8),
        QuerySpec(query="MCP tool registration", query_type=QueryType.SEMANTIC, importance=0.95),
        QuerySpec(query="mcp_server.cpp", query_type=QueryType.GET, importance=0.7),
    ]

    await planner.execute(specs, allow_adaptive=False)

    call_types = [qt for qt, _ in yams.calls]
    first_graph_idx = call_types.index(QueryType.GRAPH)
    first_grep_idx = call_types.index(QueryType.GREP)
    first_get_idx = call_types.index(QueryType.GET)

    assert call_types[0] == QueryType.SEMANTIC
    assert first_graph_idx < first_grep_idx
    assert first_graph_idx < first_get_idx


@pytest.mark.asyncio
async def test_execute_adds_graph_guided_grep_and_get_specs() -> None:
    yams = _StageStubYAMS()
    planner = QueryPlanner(yams)
    specs = [
        QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.8),
        QuerySpec(query="MCP tool registration", query_type=QueryType.SEMANTIC, importance=0.95),
        QuerySpec(query="mcp_server.cpp", query_type=QueryType.GET, importance=0.7),
    ]

    await planner.execute(specs, allow_adaptive=False)

    grep_queries = [q for qt, q in yams.calls if qt == QueryType.GREP]
    get_queries = [q for qt, q in yams.calls if qt == QueryType.GET]

    assert any("path:/repo/src/mcp/mcp_server.cpp" in q for q in grep_queries)
    assert "/repo/src/mcp/mcp_server.cpp" in get_queries


def test_top_graph_paths_prefers_related_neighbors() -> None:
    planner = QueryPlanner(_StubYAMS())
    graph_results = [
        YAMSQueryResult(
            spec=QuerySpec(
                query="/repo/src/mcp/mcp_server.cpp depth:1 limit:25",
                query_type=QueryType.GRAPH,
                importance=0.8,
            ),
            chunks=[
                YAMSChunk(
                    chunk_id="same-subsystem",
                    content="[file] /repo/include/yams/mcp/tool_registry.h",
                    source="/repo/include/yams/mcp/tool_registry.h",
                    score=0.7,
                ),
                YAMSChunk(
                    chunk_id="far-away",
                    content="[file] /repo/src/vector/embedding_service.cpp",
                    source="/repo/src/vector/embedding_service.cpp",
                    score=0.72,
                ),
            ],
        )
    ]

    ranked = planner._top_graph_paths(graph_results, limit=2)
    assert ranked[0] == "/repo/include/yams/mcp/tool_registry.h"


@pytest.mark.asyncio
async def test_maybe_dspy_rerank_reorders_top_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import dcs.planner as planner_mod

    class _FakeDSPy:
        last_context = None

        class Signature:
            pass

        class InputField:
            def __init__(self, **kwargs):
                pass

        class OutputField:
            def __init__(self, **kwargs):
                pass

        class ChatAdapter:
            pass

        class JSONAdapter:
            pass

        class _Context:
            def __init__(self, **kwargs):
                _FakeDSPy.last_context = kwargs

            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def context(**kwargs):
            return _FakeDSPy._Context(**kwargs)

        @staticmethod
        def Predict(sig):
            return _FakeDSPyPredictor()

    monkeypatch.setattr(planner_mod, "dspy", _FakeDSPy)
    planner = QueryPlanner(
        _StubYAMS(),
        dspy_rerank_model=object(),
        dspy_rerank_top_k=2,
        dspy_rerank_demos=[
            {"query": "demo", "max_ranked_ids": 2, "candidates_json": "[]", "ranked_ids": [1]}
        ],
    )
    spec = QuerySpec(query="EmbeddingService", query_type=QueryType.SEMANTIC, importance=0.9)
    chunks = [
        YAMSChunk(chunk_id="1", content="one", score=0.9, source="/repo/src/a.cpp"),
        YAMSChunk(chunk_id="2", content="two", score=0.8, source="/repo/src/b.cpp"),
    ]
    ranked = await planner._maybe_dspy_rerank(spec, chunks)
    assert ranked[0].source == "/repo/src/b.cpp"
    assert type(_FakeDSPy.last_context["adapter"]).__name__ == "JSONAdapter"
    assert _FakeDSPyPredictor.last_demos and _FakeDSPyPredictor.last_demos[0]["query"] == "demo"
