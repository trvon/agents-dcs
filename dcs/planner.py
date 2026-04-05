from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Protocol

try:  # optional dependency
    import dspy  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from dcs.shared import is_noise_source, spec_key
from dcs.types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult

logger = logging.getLogger(__name__)


class YAMSClientLike(Protocol):
    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[YAMSChunk]: ...

    async def grep(self, pattern: str, **kwargs: Any) -> list[YAMSChunk]: ...

    async def graph(self, query: str) -> list[YAMSChunk]: ...

    async def get(self, name_or_hash: str) -> YAMSChunk | None: ...

    async def list_docs(self, **kwargs: Any) -> list[YAMSChunk]: ...

    async def execute_spec(self, spec: QuerySpec) -> YAMSQueryResult: ...


_PATH_RE = re.compile(r"(?P<path>(?:[A-Za-z]:\\)?(?:\.?\.?/)?[\w.\-~/]+(?:/[\w.\-]+)+)")
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_QUERY_STOP_TERMS = {
    "path",
    "include",
    "exclude",
    "depth",
    "limit",
    "file",
    "code",
    "class",
    "method",
    "function",
    "implementation",
    "details",
    "describe",
    "explain",
    "list",
    "show",
}
_PATH_STOP_TERMS = {
    "repo",
    "src",
    "include",
    "lib",
    "tests",
    "test",
    "docs",
    "doc",
    "benchmarks",
    "benchmark",
    "yams",
    "cpp",
    "hpp",
    "h",
    "cc",
    "cxx",
    "py",
    "md",
    "json",
    "yaml",
    "yml",
}


def _dedupe_chunks(chunks: list[YAMSChunk]) -> list[YAMSChunk]:
    seen: set[str] = set()
    out: list[YAMSChunk] = []
    for c in chunks:
        cid = (c.chunk_id or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


class QueryPlanner:
    """Maps QuerySpecs to concrete YAMS executions (including expansions)."""

    def __init__(
        self,
        yams: YAMSClientLike,
        max_concurrency: int = 0,
        *,
        dspy_rerank_model: Any | None = None,
        dspy_rerank_predictor: Any | None = None,
        dspy_rerank_top_k: int = 5,
        dspy_rerank_demos: list[dict[str, Any]] | None = None,
        dspy_rerank_prefer_json: bool = True,
    ):
        self._yams = yams
        self._max_concurrency = max(0, int(max_concurrency))
        self._dspy_rerank_model = dspy_rerank_model
        self._dspy_rerank_predictor = dspy_rerank_predictor
        self._dspy_rerank_top_k = max(0, int(dspy_rerank_top_k))
        self._dspy_rerank_demos = list(dspy_rerank_demos or [])
        self._dspy_rerank_prefer_json = bool(dspy_rerank_prefer_json)

    @staticmethod
    def _build_dspy_signature() -> Any:
        class RetrievalRerankSig(dspy.Signature):  # type: ignore[misc]
            """Rank candidate files by relevance to the query.

            Prefer source files whose path and preview align with the query intent.
            Penalize docs, tests, benchmarks, and result artifacts unless the query explicitly asks for them.
            Return only the candidate ids ordered best to worst.
            """

            query: str = dspy.InputField(desc="Original retrieval query")
            max_ranked_ids: int = dspy.InputField(desc="Maximum number of candidate ids to return")
            candidates_json: str = dspy.InputField(
                desc="Compact JSON list of candidate files with ids, paths, and previews"
            )
            ranked_ids: list[int] = dspy.OutputField(
                desc="Return only candidate ids as a JSON-style integer list, best to worst, with no prose"
            )

        return RetrievalRerankSig

    @staticmethod
    def _coerce_ranked_ids(raw: Any, limit: int) -> list[int]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = [int(tok) for tok in re.findall(r"\d+", raw)]
        elif isinstance(raw, tuple):
            raw = list(raw)

        if not isinstance(raw, list):
            return []

        order: list[int] = []
        for item in raw:
            try:
                val = int(item)
            except Exception:
                continue
            if 1 <= val <= limit and val not in order:
                order.append(val)
        return order

    def _dspy_adapters(self) -> list[Any]:
        adapters: list[Any] = []
        if self._dspy_rerank_prefer_json and hasattr(dspy, "JSONAdapter"):
            adapters.append(dspy.JSONAdapter())
        if hasattr(dspy, "ChatAdapter"):
            adapters.append(dspy.ChatAdapter())
        return adapters

    async def execute(
        self, specs: list[QuerySpec], *, allow_adaptive: bool = True
    ) -> list[YAMSQueryResult]:
        """Execute query specs in staged order with graph-guided retrieval.

        Retrieval order is intentionally accuracy-first:
          1) semantic/list (search)
          2) graph (explicit + search-driven fanout)
          3) grep/get (augmented with graph path hints)
        """
        specs = self._dedupe_specs(specs)
        if not specs:
            return []

        t0 = time.perf_counter()
        search_specs, graph_specs, grep_get_specs, other_specs = self._partition_specs(specs)

        stage_results: list[YAMSQueryResult] = []

        # Stage 1: lexical/semantic discovery.
        search_results = await self._execute_specs_batch(search_specs)
        stage_results.extend(search_results)

        # Stage 2: graph traversal from explicit graph queries and top search anchors.
        graph_fanout = self._graph_fanout_from_results(search_results)
        stage2_graph_specs = self._dedupe_specs(list(graph_specs) + graph_fanout)
        graph_results = await self._execute_specs_batch(stage2_graph_specs)
        stage_results.extend(graph_results)

        # Stage 3: targeted grep/get with graph-guided file hints.
        guided_specs = self._apply_graph_guidance(grep_get_specs, graph_results)
        guided_results = await self._execute_specs_batch(self._dedupe_specs(guided_specs))
        stage_results.extend(guided_results)

        # Stage 4: any residual query types.
        tail_results = await self._execute_specs_batch(other_specs)
        stage_results.extend(tail_results)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "executed staged retrieval specs=%d search=%d graph=%d grep_get=%d other=%d in %dms",
            len(specs),
            len(search_specs),
            len(stage2_graph_specs),
            len(self._dedupe_specs(guided_specs)),
            len(other_specs),
            int(dt_ms),
        )
        merged = self._merge_results_by_spec(stage_results)

        # Adaptive followups based on actual YAMS output quality.
        follow_specs = self._adaptive_followups(merged) if allow_adaptive else []
        if follow_specs:
            follow_results = await self.execute(
                self._dedupe_specs(follow_specs), allow_adaptive=False
            )
            return self._merge_results_by_spec(merged + follow_results)

        return merged

    async def _execute_specs_batch(self, specs: list[QuerySpec]) -> list[YAMSQueryResult]:
        if not specs:
            return []

        if self._max_concurrency <= 0 or self._max_concurrency >= len(specs):
            tasks = [self._timed_execute_spec(s) for s in specs]
            return await asyncio.gather(*tasks)

        sem = asyncio.Semaphore(self._max_concurrency)

        async def run_one(spec: QuerySpec) -> YAMSQueryResult:
            async with sem:
                return await self._timed_execute_spec(spec)

        tasks = [run_one(s) for s in specs]
        return await asyncio.gather(*tasks)

    def _partition_specs(
        self, specs: list[QuerySpec]
    ) -> tuple[list[QuerySpec], list[QuerySpec], list[QuerySpec], list[QuerySpec]]:
        search_specs: list[QuerySpec] = []
        graph_specs: list[QuerySpec] = []
        grep_get_specs: list[QuerySpec] = []
        other_specs: list[QuerySpec] = []

        for s in specs:
            if s.query_type in {QueryType.SEMANTIC, QueryType.LIST}:
                search_specs.append(s)
            elif s.query_type == QueryType.GRAPH:
                graph_specs.append(s)
            elif s.query_type in {QueryType.GREP, QueryType.GET}:
                grep_get_specs.append(s)
            else:
                other_specs.append(s)

        return search_specs, graph_specs, grep_get_specs, other_specs

    def _graph_fanout_from_results(self, results: list[YAMSQueryResult]) -> list[QuerySpec]:
        out: list[QuerySpec] = []
        max_graph_specs = 4

        for res in results:
            if len(out) >= max_graph_specs:
                break
            if res.spec.query_type not in {QueryType.SEMANTIC, QueryType.LIST}:
                continue

            validated = self._validated_paths_from_results([res], min_confidence=0.58, per_result=1)
            for src in validated:
                if len(out) >= max_graph_specs:
                    break
                out.append(
                    QuerySpec(
                        query=f"{src} depth:1 limit:25",
                        query_type=QueryType.GRAPH,
                        importance=max(0.5, float(res.spec.importance) - 0.1),
                        reason="stage: graph fanout from validated search anchor",
                    )
                )

        return out

    def _apply_graph_guidance(
        self,
        specs: list[QuerySpec],
        graph_results: list[YAMSQueryResult],
    ) -> list[QuerySpec]:
        if not specs:
            return []

        graph_paths = self._top_graph_paths(graph_results, limit=8)
        if not graph_paths:
            return specs

        out = list(specs)
        for spec in specs:
            if spec.query_type == QueryType.GREP:
                q = (spec.query or "").strip()
                if not q or "path:" in q:
                    continue
                for p in self._select_paths_for_query(q, graph_paths, max_paths=1):
                    out.append(
                        QuerySpec(
                            query=f"{q} path:{p}",
                            query_type=QueryType.GREP,
                            importance=min(1.0, float(spec.importance) + 0.05),
                            reason=self._append_reason(spec.reason, "graph-guided path"),
                        )
                    )
            elif spec.query_type == QueryType.GET:
                q = (spec.query or "").strip()
                if not q:
                    continue
                if "/" in q or "\\" in q:
                    continue
                for p in self._select_paths_for_get(q, graph_paths, max_paths=2):
                    out.append(
                        QuerySpec(
                            query=p,
                            query_type=QueryType.GET,
                            importance=max(0.9, float(spec.importance)),
                            reason=self._append_reason(spec.reason, "graph-guided path"),
                        )
                    )

        return out

    async def execute_multihop(
        self, specs: list[QuerySpec], depth: int = 2
    ) -> list[YAMSQueryResult]:
        """Execute specs, then follow up on high-importance results."""
        depth = max(1, int(depth))
        primary = await self.execute(specs)
        all_results = list(primary)

        # Breadth-first expansion driven by result contents.
        visited_specs = {spec_key(r.spec) for r in all_results}
        frontier = list(primary)

        for hop in range(1, depth + 1):
            follow_specs: list[QuerySpec] = []
            for res in frontier:
                if res.spec.importance < 0.7:
                    continue
                follow_specs.extend(self._followups_from_result(res, hop=hop))

            follow_specs = [
                s for s in self._dedupe_specs(follow_specs) if spec_key(s) not in visited_specs
            ]
            if not follow_specs:
                break

            logger.debug("multihop hop=%d follow_specs=%d", hop, len(follow_specs))
            follow_results = await self.execute(follow_specs)
            for r in follow_results:
                visited_specs.add(spec_key(r.spec))

            all_results.extend(follow_results)
            frontier = follow_results

        return self._merge_results_by_spec(all_results)

    async def execute_with_expansion(self, specs: list[QuerySpec]) -> list[YAMSQueryResult]:
        """Execute specs, then graph-traverse top chunks to pull related chunks."""
        primary = await self.execute(specs)
        if not primary:
            return []

        graph_specs: list[QuerySpec] = []
        for res in primary:
            # Prefer top scored chunks; fall back to first.
            chunks = sorted(res.chunks or [], key=lambda c: c.score, reverse=True)
            for c in chunks[:2]:
                q = (c.chunk_id or "").strip()
                if not q:
                    continue
                graph_specs.append(
                    QuerySpec(
                        query=q,
                        query_type=QueryType.GRAPH,
                        importance=max(0.4, res.spec.importance - 0.2),
                        reason=f"graph expansion from {res.spec.query_type.value}:{res.spec.query}",
                    )
                )

        expanded = await self.execute(graph_specs)
        return self._merge_results_by_spec(primary + expanded)

    async def _timed_execute_spec(self, spec: QuerySpec) -> YAMSQueryResult:
        t0 = time.perf_counter()
        try:
            res = await self._yams.execute_spec(spec)
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning("execute_spec failed (%s %r): %s", spec.query_type.value, spec.query, e)
            return YAMSQueryResult(spec=spec, chunks=[], latency_ms=dt_ms, error=str(e))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        res.latency_ms = float(res.latency_ms or dt_ms)
        res.chunks = self._rank_chunks(spec, _dedupe_chunks(res.chunks or []))
        res.chunks = await self._maybe_dspy_rerank(spec, res.chunks)
        return res

    async def _maybe_dspy_rerank(self, spec: QuerySpec, chunks: list[YAMSChunk]) -> list[YAMSChunk]:
        if self._dspy_rerank_model is None or dspy is None:
            return chunks
        if spec.query_type not in {QueryType.SEMANTIC, QueryType.GREP}:
            return chunks
        if len(chunks) < 2:
            return chunks

        top_k = min(len(chunks), max(2, self._dspy_rerank_top_k))
        head = list(chunks[:top_k])
        tail = list(chunks[top_k:])

        payload = []
        for idx, chunk in enumerate(head, start=1):
            payload.append(
                {
                    "id": idx,
                    "name": self._basename(chunk.source or ""),
                    "source": chunk.source,
                    "preview": (chunk.content or "")[:300],
                }
            )

        try:
            sig = self._build_dspy_signature()
            last_err: Exception | None = None
            pred = None
            for adapter in self._dspy_adapters():

                def _run(current_adapter: Any = adapter) -> Any:
                    with dspy.context(lm=self._dspy_rerank_model, adapter=current_adapter):
                        predictor = self._dspy_rerank_predictor
                        if predictor is None:
                            predictor = dspy.Predict(sig)
                            predictor.demos = [dict(d) for d in self._dspy_rerank_demos]
                        return predictor(
                            query=spec.query,
                            max_ranked_ids=top_k,
                            candidates_json=json.dumps(payload, ensure_ascii=True),
                        )

                try:
                    pred = await asyncio.to_thread(_run)
                    break
                except Exception as e:
                    last_err = e
                    logger.debug(
                        "DSPy retrieval rerank adapter %s failed: %s",
                        type(adapter).__name__,
                        e,
                    )
            if pred is None:
                if last_err is not None:
                    raise last_err
                return chunks

            order = self._coerce_ranked_ids(getattr(pred, "ranked_ids", []) or [], len(head))
            if not order:
                return chunks

            remaining = [i for i in range(1, len(head) + 1) if i not in order]
            ordered = [head[i - 1] for i in order + remaining]
            # preserve descending scores after rerank
            for idx, chunk in enumerate(ordered):
                chunk.score = max(0.0, min(1.0, 1.0 - (0.02 * idx)))
            return ordered + tail
        except Exception as e:  # pragma: no cover
            logger.debug("DSPy retrieval rerank failed: %s", e)
            return chunks

    def _rank_chunks(self, spec: QuerySpec, chunks: list[YAMSChunk]) -> list[YAMSChunk]:
        if not chunks:
            return []

        q_terms = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_\-]+", spec.query or "")
            if len(t) >= 3 and t.lower() not in {"path", "include", "exclude"}
        ]
        q_l = (spec.query or "").lower()
        mentions_tests = any(k in q_l for k in ("test", "benchmark", "docs"))

        ranked: list[YAMSChunk] = []
        for c in chunks:
            score = float(c.score or 0.0)
            src_l = (c.source or "").lower()
            txt_l = (c.content or "").lower()
            meta = c.metadata if isinstance(c.metadata, dict) else {}

            if spec.query_type == QueryType.GREP:
                if bool(meta.get("enriched")):
                    score += 0.08
                if bool(meta.get("structured")):
                    score += 0.06
                try:
                    fm = int(meta.get("file_matches") or 0)
                except Exception:
                    fm = 0
                if fm > 0:
                    score += min(0.12, fm / 150.0)

            if spec.query_type == QueryType.SEMANTIC:
                if "line=" in txt_l or "char=" in txt_l:
                    score += 0.06

            if q_terms:
                hits = sum(1 for t in q_terms if t in txt_l or t in src_l)
                score += min(0.15, 0.03 * hits)

            if not mentions_tests and is_noise_source(src_l):
                score *= 0.70

            c.score = max(0.0, min(1.0, score))
            ranked.append(c)

        ranked.sort(key=lambda x: (float(x.score or 0.0), x.source, x.chunk_id), reverse=True)
        return ranked

    def _dedupe_specs(self, specs: list[QuerySpec]) -> list[QuerySpec]:
        # Keep first occurrence; callers can pre-sort by importance.
        out: list[QuerySpec] = []
        seen: set[tuple[str, str]] = set()
        for s in specs or []:
            if not s or not s.query.strip():
                continue
            key = spec_key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                QuerySpec(
                    query=s.query.strip(),
                    query_type=s.query_type,
                    importance=float(max(0.0, min(1.0, s.importance))),
                    reason=s.reason or "",
                )
            )
        return out

    @staticmethod
    def _append_reason(reason: str, suffix: str) -> str:
        base = (reason or "").strip()
        if not base:
            return suffix
        if suffix in base:
            return base
        return f"{base}; {suffix}"

    @staticmethod
    def _normalize_path_candidate(raw: str) -> str | None:
        s = (raw or "").strip().strip("'\"")
        if not s:
            return None
        if s.startswith("path:file:"):
            s = s[len("path:file:") :]
        elif s.startswith("file:"):
            s = s[len("file:") :]

        # Trim structured text artifacts from graph chunk formatting.
        if " key=" in s:
            s = s.split(" key=", 1)[0].strip()
        if " (distance=" in s:
            s = s.split(" (distance=", 1)[0].strip()

        if s.startswith(("http://", "https://")):
            return None
        if "/" not in s and "\\" not in s:
            return None
        return s

    @staticmethod
    def _basename(path: str) -> str:
        return (path or "").replace("\\", "/").rsplit("/", 1)[-1]

    def _path_from_chunk(self, c: YAMSChunk) -> str | None:
        candidates: list[str] = []
        if c.source:
            candidates.append(c.source)

        meta = c.metadata if isinstance(c.metadata, dict) else {}
        for key in ("path", "file", "file_path", "label", "nodeKey", "node_key", "name"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(v)

        for cand in candidates:
            normalized = self._normalize_path_candidate(cand)
            if normalized:
                return normalized

        text = (c.content or "").strip()
        if text:
            m = _PATH_RE.search(text)
            if m:
                normalized = self._normalize_path_candidate(m.group("path"))
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _query_terms(text: str) -> list[str]:
        terms: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_\-]+", text or ""):
            t = token.lower()
            if len(t) < 3 or t in _QUERY_STOP_TERMS:
                continue
            if t not in terms:
                terms.append(t)
        return terms

    @staticmethod
    def _path_parts(path: str) -> list[str]:
        parts: list[str] = []
        for token in re.split(r"[/\\._\-]+", path or ""):
            t = token.lower().strip()
            if len(t) < 2:
                continue
            if t in _PATH_STOP_TERMS:
                continue
            parts.append(t)
        return parts

    def _anchor_confidence(self, spec: QuerySpec, chunk: YAMSChunk) -> float:
        path = self._path_from_chunk(chunk)
        if not path or is_noise_source(path):
            return 0.0

        score = float(chunk.score or 0.0)
        path_parts = self._path_parts(path)
        q_terms = self._query_terms(spec.query)
        if not q_terms:
            return max(0.0, min(1.0, score))

        overlaps = sum(1 for term in q_terms if term in path_parts or term in path.lower())
        text_l = (chunk.content or "").lower()
        text_hits = sum(1 for term in q_terms if term in text_l)
        basename = self._basename(path).lower()

        conf = 0.50 * score
        if overlaps:
            conf += min(0.30, 0.10 * overlaps)
        if text_hits:
            conf += min(0.15, 0.05 * text_hits)
        if any(term in basename for term in q_terms):
            conf += 0.08
        if "." in basename:
            conf += 0.05
        if spec.query_type == QueryType.GET:
            conf += 0.10
        if spec.query_type == QueryType.GREP and "path:" in (spec.query or ""):
            conf += 0.10
        return max(0.0, min(1.0, conf))

    def _validated_paths_from_results(
        self,
        results: list[YAMSQueryResult],
        *,
        min_confidence: float,
        per_result: int,
    ) -> list[str]:
        scored: dict[str, float] = {}
        for res in results:
            ranked = sorted(
                list(res.chunks or []),
                key=lambda c: self._anchor_confidence(res.spec, c),
                reverse=True,
            )
            emitted = 0
            for chunk in ranked:
                if emitted >= per_result:
                    break
                path = self._path_from_chunk(chunk)
                if not path:
                    continue
                conf = self._anchor_confidence(res.spec, chunk)
                if conf < float(min_confidence):
                    continue
                scored[path] = max(scored.get(path, 0.0), conf)
                emitted += 1

        ordered = sorted(scored.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [path for path, _ in ordered]

    def _top_graph_paths(self, graph_results: list[YAMSQueryResult], limit: int = 10) -> list[str]:
        scored: dict[str, float] = {}
        for res in graph_results:
            anchor_hint = (res.spec.query or "").split(" depth:", 1)[0].strip()
            anchor_parts = set(self._path_parts(anchor_hint))
            for c in res.chunks or []:
                p = self._path_from_chunk(c)
                if not p:
                    continue
                if is_noise_source(p):
                    continue
                path_parts = set(self._path_parts(p))
                overlap = len(anchor_parts.intersection(path_parts)) if anchor_parts else 0
                score = float(c.score or 0.0) + min(0.18, 0.06 * overlap)
                if anchor_hint and p == anchor_hint:
                    score -= 0.05
                if anchor_parts and overlap == 0:
                    score *= 0.75
                scored[p] = max(scored.get(p, 0.0), score)

        ordered = sorted(scored.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [p for p, _ in ordered[: max(1, int(limit))]]

    def _select_paths_for_query(
        self,
        query: str,
        paths: list[str],
        *,
        max_paths: int,
    ) -> list[str]:
        if not paths:
            return []
        q_terms = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_\-]+", query or "")
            if len(t) >= 3 and t.lower() not in {"path", "include", "exclude"}
        ]
        if not q_terms:
            return paths[:max_paths]

        scored: list[tuple[int, str]] = []
        for p in paths:
            pl = p.lower()
            base = self._basename(pl)
            score = sum(1 for t in q_terms if t in pl)
            if any(t in base for t in q_terms):
                score += 1
            scored.append((score, p))

        scored.sort(key=lambda it: (it[0], it[1]), reverse=True)
        top = [p for s, p in scored if s > 0][:max_paths]
        if top:
            return top
        return paths[:max_paths]

    def _select_paths_for_get(
        self,
        query: str,
        paths: list[str],
        *,
        max_paths: int,
    ) -> list[str]:
        q = (query or "").strip().lower()
        if not q:
            return []

        exact = [p for p in paths if self._basename(p).lower() == q]
        if exact:
            return exact[:max_paths]

        partial = [p for p in paths if q in self._basename(p).lower()]
        if partial:
            return partial[:max_paths]

        return self._select_paths_for_query(query, paths, max_paths=max_paths)

    def _merge_results_by_spec(self, results: list[YAMSQueryResult]) -> list[YAMSQueryResult]:
        merged: dict[tuple[str, str], YAMSQueryResult] = {}
        order: list[tuple[str, str]] = []

        for r in results or []:
            key = spec_key(r.spec)
            if key not in merged:
                merged[key] = YAMSQueryResult(
                    spec=r.spec,
                    chunks=_dedupe_chunks(list(r.chunks or [])),
                    latency_ms=float(r.latency_ms or 0.0),
                    error=r.error,
                )
                order.append(key)
                continue

            cur = merged[key]
            cur.chunks = _dedupe_chunks(cur.chunks + list(r.chunks or []))
            cur.latency_ms = max(float(cur.latency_ms or 0.0), float(r.latency_ms or 0.0))
            if cur.error is None and r.error:
                cur.error = r.error

        return [merged[k] for k in order]

    def _followups_from_result(self, res: YAMSQueryResult, hop: int) -> list[QuerySpec]:
        text = "\n".join(c.content for c in (res.chunks or []) if c.content)
        if not text:
            return []

        follow: list[QuerySpec] = []
        # Files mentioned -> GET
        paths: set[str] = set()
        for m in _PATH_RE.finditer(text):
            p = m.group("path").strip().strip("'\"")
            if "/" not in p:
                continue
            # Avoid obvious URLs.
            if p.startswith("http://") or p.startswith("https://"):
                continue
            paths.add(p)

        for p in list(sorted(paths))[:5]:
            follow.append(
                QuerySpec(
                    query=p,
                    query_type=QueryType.GET,
                    importance=max(0.4, res.spec.importance - 0.1),
                    reason=f"follow-up hop {hop}: file mentioned",
                )
            )

        # Function/class identifiers -> SEMANTIC
        idents: set[str] = set()
        for m in _IDENT_RE.finditer(text):
            ident = m.group(1)
            if len(ident) < 3:
                continue
            if ident in {"self", "None", "True", "False"}:
                continue
            idents.add(ident)

        for ident in list(sorted(idents))[:5]:
            follow.append(
                QuerySpec(
                    query=ident,
                    query_type=QueryType.SEMANTIC,
                    importance=max(0.3, res.spec.importance - 0.25),
                    reason=f"follow-up hop {hop}: symbol mentioned",
                )
            )

        return follow

    def _adaptive_followups(self, results: list[YAMSQueryResult]) -> list[QuerySpec]:
        follow: list[QuerySpec] = []
        max_followups = 5

        for res in results:
            if len(follow) >= max_followups:
                break

            chunks = res.chunks or []
            empty = len(chunks) == 0
            path_only = all((c.metadata or {}).get("enriched") is False for c in chunks)
            is_grep = res.spec.query_type == QueryType.GREP
            is_semantic = res.spec.query_type == QueryType.SEMANTIC
            is_get = res.spec.query_type == QueryType.GET

            # GREP results empty or path-only -> try GET on referenced paths
            if is_grep and (empty or path_only):
                hint_path = None
                if "path:" in res.spec.query:
                    hint_path = res.spec.query.split("path:", 1)[1].strip().strip("'\"")
                if hint_path:
                    follow.append(
                        QuerySpec(
                            query=hint_path,
                            query_type=QueryType.GET,
                            importance=max(0.6, res.spec.importance - 0.1),
                            reason="adaptive: grep path-only -> get",
                        )
                    )
                    continue

                for c in chunks[:3]:
                    p = self._path_from_chunk(c)
                    if p:
                        follow.append(
                            QuerySpec(
                                query=p,
                                query_type=QueryType.GET,
                                importance=max(0.6, res.spec.importance - 0.1),
                                reason="adaptive: grep path-only -> get",
                            )
                        )
                    if len(follow) >= max_followups:
                        break
                continue

            # SEMANTIC with no hits -> try GREP on identifiers
            if is_semantic and empty:
                idents = [m.group(1) for m in _IDENT_RE.finditer(res.spec.query or "")]
                for ident in idents[:3]:
                    follow.append(
                        QuerySpec(
                            query=ident,
                            query_type=QueryType.GREP,
                            importance=max(0.5, res.spec.importance - 0.2),
                            reason="adaptive: semantic empty -> grep ident",
                        )
                    )
                    if len(follow) >= max_followups:
                        break
                continue

            # GET failed -> try GREP on basename
            if is_get and empty:
                base = (res.spec.query or "").split("/")[-1]
                if base:
                    follow.append(
                        QuerySpec(
                            query=base,
                            query_type=QueryType.GREP,
                            importance=max(0.5, res.spec.importance - 0.2),
                            reason="adaptive: get empty -> grep basename",
                        )
                    )

            # High-signal file-based chunks -> deterministic graph expansion.
            # This improves relation discovery for architecture/knowledge-graph tasks.
            if len(follow) >= max_followups:
                continue

            if (
                res.spec.query_type in {QueryType.GREP, QueryType.GET, QueryType.SEMANTIC}
                and not empty
            ):
                # Prefer top scored non-noise sources.
                ranked = sorted(chunks, key=lambda c: float(c.score or 0.0), reverse=True)
                emitted = 0
                for c in ranked:
                    src = self._path_from_chunk(c) or (c.source or "")
                    src = (src or "").strip()
                    if not src:
                        continue
                    if is_noise_source(src):
                        continue
                    if "/" not in src and "\\" not in src:
                        continue
                    if self._anchor_confidence(res.spec, c) < 0.55:
                        continue
                    follow.append(
                        QuerySpec(
                            query=f"{src} depth:1 limit:25",
                            query_type=QueryType.GRAPH,
                            importance=max(0.45, res.spec.importance - 0.15),
                            reason="adaptive: validated file anchor -> graph expansion",
                        )
                    )
                    emitted += 1
                    if emitted >= 1 or len(follow) >= max_followups:
                        break

        return follow
