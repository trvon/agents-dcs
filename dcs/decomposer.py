from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
from dataclasses import asdict
from typing import Any

from dcs.shared import clamp01, spec_key
from dcs.types import Critique, ModelConfig, QuerySpec, QueryType

logger = logging.getLogger(__name__)


_PATH_RE = re.compile(r"(?P<path>(?:[A-Za-z]:\\)?(?:\.?\.?/)?[\w.\-~/]+(?:/[\w.\-]+)+)")
_PY_SYMBOL_RE = re.compile(r"\b(?:(?:class|def)\s+)?([A-Za-z_][A-Za-z0-9_]*)\b")
_ERROR_RE = re.compile(r"(?i)\b(?:traceback|exception|error|failed|assert(?:ion)?\s+error|panic)\b")


class TaskDecomposer:
    """Breaks a task into structured YAMS information needs."""

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            openai_mod = importlib.import_module("openai")
            async_openai_cls = openai_mod.AsyncOpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "openai package not available; install `openai` to use TaskDecomposer"
            ) from e

        self._client: Any = async_openai_cls(base_url=config.base_url, api_key=config.api_key)

    async def decompose(
        self,
        task: str,
        max_queries: int = 5,
        type_bias: dict[str, float] | None = None,
        require_types: set[QueryType] | None = None,
        use_task_seeding: bool = True,
    ) -> list[QuerySpec]:
        """Use the model to propose QuerySpecs; fall back heuristically."""
        task = (task or "").strip()
        if not task:
            return []

        specs: list[QuerySpec] = []
        try:
            specs = await self._model_decompose(task=task, max_queries=max_queries)
        except Exception as e:
            logger.warning("decompose: model failed, using fallback: %s", e)

        if not specs:
            specs = self._fallback_decompose(task=task, max_queries=max_queries)

        specs = self._apply_path_hints(task, specs)
        if use_task_seeding:
            specs = self._inject_task_specific_specs(task, specs)
        specs = self._apply_type_bias(specs, type_bias)
        specs = self._ensure_required_types(task, specs, require_types)
        return self._postprocess_specs(specs, max_queries=max_queries)

    async def refine(
        self,
        task: str,
        critique: Critique,
        previous_specs: list[QuerySpec],
        *,
        type_bias: dict[str, float] | None = None,
        require_types: set[QueryType] | None = None,
        use_task_seeding: bool = True,
    ) -> list[QuerySpec]:
        """Refine/additional queries based on critique; avoid repeats."""
        task = (task or "").strip()
        if not task:
            return []

        tried = {spec_key(s) for s in (previous_specs or [])}

        # If critique is empty, do not spin cycles.
        if not (critique.missing_info or critique.suggested_queries):
            logger.info("refine: no missing_info or suggested_queries; skipping")
            return []

        logger.info(
            "refine: missing_info=%r, suggested_queries=%r",
            critique.missing_info,
            critique.suggested_queries,
        )

        try:
            specs = await self._model_refine(task, critique, previous_specs)
            logger.info(
                "refine: model returned %d specs: %r",
                len(specs),
                [(s.query_type.value, s.query) for s in specs],
            )
        except Exception as e:
            logger.warning("refine: model failed, using fallback: %s", e)
            specs = []

        if not specs:
            logger.info("refine: using heuristic fallback from critique")
            # Heuristic refinement: turn missing_info and suggested_queries
            # into GREP specs with cleaned, executable patterns.
            specs = []

            for mi in critique.missing_info:
                # First, extract any mentioned filenames and emit targeted specs
                file_specs = self._extract_file_grep_specs(mi)
                specs.extend(file_specs)

                q = self._clean_query_for_grep(mi)
                if q:
                    specs.append(
                        QuerySpec(
                            query=q,
                            query_type=QueryType.GREP,
                            importance=0.7,
                            reason="missing_info",
                        )
                    )

            for sq in critique.suggested_queries:
                # Extract file names first
                file_specs = self._extract_file_grep_specs(sq)
                specs.extend(file_specs)

                q = self._clean_query_for_grep(sq)
                if q:
                    specs.append(
                        QuerySpec(
                            query=q,
                            query_type=QueryType.GREP,
                            importance=0.6,
                            reason="suggested_query",
                        )
                    )

            # Also extract task-derived patterns — these catch domain-specific
            # identifiers (e.g. "tools registered" -> registerTool) that the
            # critique text doesn't mention but are needed for targeted retrieval.
            task_specs = self._extract_task_grep_specs(task)
            specs.extend(task_specs)

        specs = self._apply_path_hints(task, specs)
        if use_task_seeding:
            specs = self._inject_task_specific_specs(task, specs)

        # Deduplicate against previous and within new.
        out: list[QuerySpec] = []
        seen = set(tried)
        for spec in specs:
            key = spec_key(spec)
            if key in seen:
                logger.debug(
                    "refine: skipping duplicate spec: %s %r", spec.query_type.value, spec.query
                )
                continue
            seen.add(key)
            out.append(spec)

        out = self._apply_type_bias(out, type_bias)
        out = self._ensure_required_types(task, out, require_types)
        out = self._postprocess_specs(out, max_queries=5)

        logger.info(
            "refine: emitting %d new specs: %r",
            len(out),
            [(s.query_type.value, s.query) for s in out],
        )
        return out

    def _inject_task_specific_specs(self, task: str, specs: list[QuerySpec]) -> list[QuerySpec]:
        """Add high-signal task-specific retrieval specs for known hard domains."""
        task_l = (task or "").lower()
        is_mcp_task = "mcp" in task_l or "model context protocol" in task_l
        is_embedding_task = "embeddingservice" in task_l or "embedding service" in task_l
        is_kg_task = "knowledge graph" in task_l or "knowledge-graph" in task_l
        is_event_bus_task = "event bus" in task_l or "internaleventbus" in task_l
        is_daemon_arch_task = "daemon architecture" in task_l or (
            "major components" in task_l and "daemon" in task_l
        )
        is_storage_task = "store documents" in task_l or "content-addressable" in task_l
        is_hybrid_search_task = "hybrid search" in task_l and "search" in task_l

        if not (
            is_mcp_task
            or is_embedding_task
            or is_kg_task
            or is_event_bus_task
            or is_daemon_arch_task
            or is_storage_task
            or is_hybrid_search_task
        ):
            return specs

        seeded: list[QuerySpec] = []

        if is_mcp_task:
            is_tools_task = "tool" in task_l and ("register" in task_l or "registered" in task_l)
            is_protocol_task = any(
                k in task_l
                for k in (
                    "transport",
                    "stdio",
                    "ndjson",
                    "json-rpc",
                    "jsonrpc",
                    "communicate",
                    "server started",
                    "serve",
                )
            )

            if is_tools_task:
                seeded.extend(
                    [
                        QuerySpec(
                            query="registerTool",
                            query_type=QueryType.GREP,
                            importance=0.99,
                            reason="mcp tool registration anchors",
                        ),
                        QuerySpec(
                            query="registerRawTool",
                            query_type=QueryType.GREP,
                            importance=0.98,
                            reason="mcp raw tool registration anchors",
                        ),
                        QuerySpec(
                            query="Store documents",
                            query_type=QueryType.GREP,
                            importance=0.97,
                            reason="tool description includes store wording",
                        ),
                        QuerySpec(
                            query="session_start",
                            query_type=QueryType.GREP,
                            importance=0.96,
                            reason="session tool registration string",
                        ),
                        QuerySpec(
                            query="updateMetadata|update_metadata",
                            query_type=QueryType.GREP,
                            importance=0.95,
                            reason="metadata tool registration string",
                        ),
                    ]
                )

            if is_protocol_task:
                seeded.extend(
                    [
                        QuerySpec(
                            query="ndjson",
                            query_type=QueryType.GREP,
                            importance=0.99,
                            reason="transport framing keyword",
                        ),
                        QuerySpec(
                            query="json-rpc",
                            query_type=QueryType.GREP,
                            importance=0.98,
                            reason="rpc protocol keyword",
                        ),
                        QuerySpec(
                            query="stdio_transport",
                            query_type=QueryType.GREP,
                            importance=0.97,
                            reason="transport implementation symbol",
                        ),
                        QuerySpec(
                            query="serve_command",
                            query_type=QueryType.GREP,
                            importance=0.96,
                            reason="server startup command implementation",
                        ),
                        QuerySpec(
                            query="registerTool",
                            query_type=QueryType.GREP,
                            importance=0.95,
                            reason="tool registration API",
                        ),
                    ]
                )

            if not (is_tools_task or is_protocol_task):
                seeded.extend(
                    [
                        QuerySpec(
                            query="registerTool",
                            query_type=QueryType.GREP,
                            importance=0.98,
                            reason="default mcp tool registration anchor",
                        ),
                        QuerySpec(
                            query="ndjson",
                            query_type=QueryType.GREP,
                            importance=0.95,
                            reason="default mcp transport anchor",
                        ),
                    ]
                )

        if is_embedding_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="src/vector/embedding_service.cpp",
                        query_type=QueryType.GET,
                        importance=0.99,
                        reason="primary embedding implementation",
                    ),
                    QuerySpec(
                        query="EmbeddingService",
                        query_type=QueryType.GREP,
                        importance=0.98,
                        reason="embedding service class anchor",
                    ),
                    QuerySpec(
                        query="all-MiniLM-L6-v2",
                        query_type=QueryType.GREP,
                        importance=0.97,
                        reason="default embedding model and dimensions",
                    ),
                    QuerySpec(
                        query="include/yams/vector/dim_resolver.h",
                        query_type=QueryType.GET,
                        importance=0.96,
                        reason="model-name defaults and dimensions",
                    ),
                    QuerySpec(
                        query="generateEmbeddingsInternal",
                        query_type=QueryType.GREP,
                        importance=0.95,
                        reason="batching and queue mechanics",
                    ),
                    QuerySpec(
                        query="vectors.db",
                        query_type=QueryType.GREP,
                        importance=0.94,
                        reason="embedding storage backend details",
                    ),
                    QuerySpec(
                        query="include/yams/vector/embedding_service.h",
                        query_type=QueryType.GET,
                        importance=0.9,
                        reason="embedding service API surface",
                    ),
                ]
            )

        if is_kg_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="KnowledgeGraphStore|knowledge graph|KG",
                        query_type=QueryType.GREP,
                        importance=0.99,
                        reason="knowledge graph component anchors",
                    ),
                    QuerySpec(
                        query="node|edge|relation path:src/metadata/knowledge_graph_store_sqlite.cpp",
                        query_type=QueryType.GREP,
                        importance=0.98,
                        reason="graph structure representation",
                    ),
                    QuerySpec(
                        query="search|query|travers path:src/metadata/knowledge_graph_store_sqlite.cpp",
                        query_type=QueryType.GREP,
                        importance=0.97,
                        reason="search-time graph usage",
                    ),
                    QuerySpec(
                        query="src/metadata/knowledge_graph_store_sqlite.cpp",
                        query_type=QueryType.GET,
                        importance=0.96,
                        reason="knowledge graph storage implementation",
                    ),
                    QuerySpec(
                        query="src/metadata/knowledge_graph_store_sqlite.cpp depth:1 limit:40",
                        query_type=QueryType.GRAPH,
                        importance=0.95,
                        reason="knowledge graph relation expansion from storage implementation",
                    ),
                    QuerySpec(
                        query="include/yams/metadata/knowledge_graph_store.h",
                        query_type=QueryType.GET,
                        importance=0.9,
                        reason="knowledge graph API",
                    ),
                ]
            )

        if is_event_bus_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="include/yams/daemon/components/InternalEventBus.h",
                        query_type=QueryType.GET,
                        importance=0.99,
                        reason="event bus type/channel definitions",
                    ),
                    QuerySpec(
                        query="InternalEventBus path:include/yams/daemon/components/InternalEventBus.h",
                        query_type=QueryType.GREP,
                        importance=0.97,
                        reason="event bus symbol anchored to primary header",
                    ),
                    QuerySpec(
                        query="get_or_create_channel|try_push|try_pop path:include/yams/daemon/components/InternalEventBus.h",
                        query_type=QueryType.GREP,
                        importance=0.96,
                        reason="channel keying and publish/subscribe operations",
                    ),
                    QuerySpec(
                        query="src/daemon/components/PostIngestQueue.cpp",
                        query_type=QueryType.GET,
                        importance=0.92,
                        reason="component usage of event bus channels",
                    ),
                ]
            )

        if is_daemon_arch_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="include/yams/daemon/components/ServiceManager.h",
                        query_type=QueryType.GET,
                        importance=0.98,
                        reason="daemon component composition entrypoint",
                    ),
                    QuerySpec(
                        query="include/yams/daemon/components/InternalEventBus.h",
                        query_type=QueryType.GET,
                        importance=0.96,
                        reason="event bus architecture role",
                    ),
                    QuerySpec(
                        query="include/yams/daemon/components/ResourceGovernor.h",
                        query_type=QueryType.GET,
                        importance=0.95,
                        reason="resource pressure and admission role",
                    ),
                    QuerySpec(
                        query="ServiceManager|InternalEventBus|ResourceGovernor|EmbeddingService",
                        query_type=QueryType.GREP,
                        importance=0.94,
                        reason="major daemon components for architecture summary",
                    ),
                ]
            )

        if is_storage_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="include/yams/api/content_store.h",
                        query_type=QueryType.GET,
                        importance=0.98,
                        reason="content store API and CAS model",
                    ),
                    QuerySpec(
                        query="src/api/content_store_impl.cpp",
                        query_type=QueryType.GET,
                        importance=0.97,
                        reason="content store implementation details",
                    ),
                    QuerySpec(
                        query="src/metadata/metadata_repository.cpp",
                        query_type=QueryType.GET,
                        importance=0.96,
                        reason="metadata/index persistence path",
                    ),
                    QuerySpec(
                        query="content-addressable|sha256|metadata|index",
                        query_type=QueryType.GREP,
                        importance=0.95,
                        reason="core storage semantics and indexing",
                    ),
                ]
            )

        if is_hybrid_search_task:
            seeded.extend(
                [
                    QuerySpec(
                        query="src/search/search_engine.cpp",
                        query_type=QueryType.GET,
                        importance=0.99,
                        reason="hybrid fusion implementation",
                    ),
                    QuerySpec(
                        query="include/yams/search/search_engine.h",
                        query_type=QueryType.GET,
                        importance=0.96,
                        reason="search backend/type declarations",
                    ),
                    QuerySpec(
                        query="hybrid|fusion|FTS5|semantic|vector",
                        query_type=QueryType.GREP,
                        importance=0.95,
                        reason="hybrid backend fusion anchors",
                    ),
                ]
            )

        if not seeded:
            seeded.extend(
                [
                    QuerySpec(
                        query="registerTool",
                        query_type=QueryType.GREP,
                        importance=0.98,
                        reason="default mcp tool registration anchor",
                    ),
                    QuerySpec(
                        query="ndjson",
                        query_type=QueryType.GREP,
                        importance=0.95,
                        reason="default mcp transport anchor",
                    ),
                ]
            )

        existing = {spec_key(s) for s in specs}
        out: list[QuerySpec] = []
        for s in seeded:
            key = spec_key(s)
            if key in existing:
                continue
            existing.add(key)
            out.append(s)

        # Prepend seeds so they survive max_queries trimming.
        return out + list(specs)

    def _apply_path_hints(self, task: str, specs: list[QuerySpec]) -> list[QuerySpec]:
        """Ensure path-hinted GREP specs when a filename is mentioned in the task."""
        filenames = self._FILENAME_RE.findall(task)
        if not filenames:
            return specs

        fname = filenames[0]
        hinted: list[QuerySpec] = []

        lowered = task.lower()
        if "tool" in lowered and ("register" in lowered or "registered" in lowered):
            hinted.append(
                QuerySpec(
                    query=f"registerTool path:{fname}",
                    query_type=QueryType.GREP,
                    importance=0.95,
                    reason=f"forced registerTool path-hinted ({fname})",
                )
            )

        grep_specs = [s for s in specs if s.query_type == QueryType.GREP]
        if grep_specs:
            ranked = sorted(grep_specs, key=lambda s: float(s.importance), reverse=True)
            for spec in ranked[:2]:
                if "path:" in spec.query:
                    continue
                hinted.append(
                    QuerySpec(
                        query=f"{spec.query} path:{fname}",
                        query_type=QueryType.GREP,
                        importance=min(1.0, float(spec.importance) + 0.1),
                        reason=f"path-hinted ({fname})",
                    )
                )
        else:
            stem = fname.rsplit(".", 1)[0]
            hinted.append(
                QuerySpec(
                    query=f"{stem} path:{fname}",
                    query_type=QueryType.GREP,
                    importance=0.9,
                    reason=f"forced path-hinted grep ({fname})",
                )
            )

        hinted_bases = {
            s.query.split(" path:", 1)[0]
            for s in hinted
            if s.query_type == QueryType.GREP and "path:" in s.query
        }
        filtered = [
            s
            for s in specs
            if not (
                s.query_type == QueryType.GREP
                and s.query in hinted_bases
                and "path:" not in s.query
            )
        ]

        # Prepend hinted specs so they survive truncation
        return hinted + filtered

    def _apply_type_bias(
        self,
        specs: list[QuerySpec],
        type_bias: dict[str, float] | None,
    ) -> list[QuerySpec]:
        if not type_bias:
            return specs

        out: list[QuerySpec] = []
        for s in specs:
            mul = float(type_bias.get(s.query_type.value, 1.0))
            out.append(
                QuerySpec(
                    query=s.query,
                    query_type=s.query_type,
                    importance=clamp01(float(s.importance) * mul),
                    reason=s.reason,
                )
            )
        return out

    def _ensure_required_types(
        self,
        task: str,
        specs: list[QuerySpec],
        required_types: set[QueryType] | None,
    ) -> list[QuerySpec]:
        if not required_types:
            return specs

        out = list(specs)
        present = {s.query_type for s in out}

        if QueryType.GREP in required_types and QueryType.GREP not in present:
            task_specs = self._extract_task_grep_specs(task)
            grep_seed = next((s for s in task_specs if s.query_type == QueryType.GREP), None)
            if grep_seed is None:
                q = self._clean_query_for_grep(task)
                if q:
                    grep_seed = QuerySpec(
                        query=q,
                        query_type=QueryType.GREP,
                        importance=0.94,
                        reason="required grep fallback",
                    )
            if grep_seed is not None:
                out.append(grep_seed)
                present.add(QueryType.GREP)

        if QueryType.GET in required_types and QueryType.GET not in present:
            path = None
            m = _PATH_RE.search(task)
            if m:
                p = m.group("path").strip().strip("'\"")
                if p:
                    path = p
            if not path:
                filenames = self._FILENAME_RE.findall(task)
                if filenames:
                    path = filenames[0]

            if path:
                out.append(
                    QuerySpec(
                        query=path,
                        query_type=QueryType.GET,
                        importance=0.92,
                        reason="required get fallback",
                    )
                )

        return out

    async def _model_decompose(self, task: str, max_queries: int) -> list[QuerySpec]:
        sys_prompt = (
            "You are a retrieval query planner for YAMS (a code+notes memory store).\n"
            "Return ONLY valid JSON: an array of query spec objects.\n"
            "Each object MUST have: query (string), query_type (one of: semantic, grep, graph, get, list), "
            "importance (number 0.0-1.0), reason (string).\n"
            "Rules:\n"
            "- Keep queries short and executable.\n"
            "- Prefer GET when a concrete path/doc id is known.\n"
            "- Prefer GREP for exact error strings, function names, constants.\n"
            "- Prefer SEMANTIC for conceptual needs.\n"
            "- Never include markdown fences.\n"
        )

        user_prompt = f"Task:\n{task}\n\nReturn up to {max_queries} query specs."

        few_shot = (
            "Example 1:\n"
            "Task: Fix failing test in src/foo.py: test_bar expects ValueError\n"
            "Output:\n"
            '[{"query":"src/foo.py","query_type":"get","importance":0.95,"reason":"inspect implementation"},'
            '{"query":"test_bar ValueError","query_type":"grep","importance":0.8,"reason":"locate failing assertion"}]\n\n'
            "Example 2:\n"
            "Task: Understand how ResourceGovernor decides canLoadModel\n"
            "Output:\n"
            '[{"query":"ResourceGovernor canLoadModel","query_type":"semantic","importance":0.9,"reason":"find policy implementation and docs"},'
            '{"query":"canLoadModel","query_type":"grep","importance":0.7,"reason":"find exact symbol references"}]'
        )

        raw = await self._chat_json(
            sys_prompt=sys_prompt, user_prompt=user_prompt, few_shot=few_shot
        )
        specs = self._parse_specs_json(raw)
        return specs

    async def _model_refine(
        self, task: str, critique: Critique, previous_specs: list[QuerySpec]
    ) -> list[QuerySpec]:
        sys_prompt = (
            "You refine retrieval queries for YAMS. Return ONLY valid JSON array of query specs."
            " Do NOT repeat previous queries. Use missing_info and suggested_queries."
        )
        prev = [
            {
                "query": s.query,
                "query_type": s.query_type.value,
                "importance": s.importance,
                "reason": s.reason,
            }
            for s in (previous_specs or [])
        ]
        user_prompt = (
            "Task:\n"
            f"{task}\n\n"
            "Critique JSON:\n"
            f"{json.dumps(asdict(critique), ensure_ascii=True)}\n\n"
            "Previous query specs JSON:\n"
            f"{json.dumps(prev, ensure_ascii=True)}\n\n"
            "Return new query specs only."
        )

        few_shot = (
            "Example:\n"
            "Task: Add caching to yams search\n"
            'Critique JSON: {"missing_info":["where queries are executed"],"suggested_queries":["execute_spec implementation"]}\n'
            'Previous query specs JSON: [{"query":"search","query_type":"semantic","importance":0.7,"reason":"initial"}]\n'
            "Output:\n"
            '[{"query":"execute_spec","query_type":"semantic","importance":0.85,"reason":"locate execution path"}]'
        )

        raw = await self._chat_json(
            sys_prompt=sys_prompt, user_prompt=user_prompt, few_shot=few_shot
        )
        specs = self._parse_specs_json(raw)
        return specs

    async def _chat_json(self, sys_prompt: str, user_prompt: str, few_shot: str) -> str:
        # Small-model-friendly: keep messages short, put examples in system.
        messages = [
            {"role": "system", "content": sys_prompt + "\n\n" + few_shot},
            {"role": "user", "content": user_prompt},
        ]

        t0 = asyncio.get_running_loop().time()
        resp = await self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        dt_ms = (asyncio.get_running_loop().time() - t0) * 1000.0

        content = (resp.choices[0].message.content or "").strip()
        logger.debug("model response (%s ms): %s", int(dt_ms), content[:500])
        return content

    def _parse_specs_json(self, text: str) -> list[QuerySpec]:
        text = (text or "").strip()
        if not text:
            return []

        # Strip <think>...</think> blocks (qwen3, deepseek-r1, etc.)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text  # fallback if stripping ate everything

        # Try markdown code fences first (```json [...] ```)
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1)
        else:
            # Some models prepend prose; attempt to extract the first JSON array.
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start : end + 1]
            else:
                candidate = cleaned

        try:
            data = json.loads(candidate)
        except Exception as e:
            logger.debug("json parse failed: %s; text=%r", e, text[:500])
            return []

        if not isinstance(data, list):
            return []

        specs: list[QuerySpec] = []
        for item in data:
            spec = self._coerce_spec(item)
            if spec is not None:
                specs.append(spec)
        return specs

    def _coerce_spec(self, item: Any) -> QuerySpec | None:
        if not isinstance(item, dict):
            return None

        query = str(item.get("query", "")).strip()
        qt_raw = str(item.get("query_type", "")).strip().lower()
        reason = str(item.get("reason", "")).strip()
        imp_raw = item.get("importance", 0.5)

        if not query or not qt_raw:
            return None

        try:
            query_type = QueryType(qt_raw)
        except Exception:
            return None

        try:
            importance = float(imp_raw)
        except Exception:
            importance = 0.5

        return QuerySpec(
            query=query,
            query_type=query_type,
            importance=clamp01(importance),
            reason=reason,
        )

    def _postprocess_specs(self, specs: list[QuerySpec], max_queries: int) -> list[QuerySpec]:
        # Deduplicate and clamp; keep highest-importance duplicates.
        best: dict[tuple[str, str], QuerySpec] = {}
        for s in specs:
            key = spec_key(s)
            prev = best.get(key)
            if prev is None or s.importance > prev.importance:
                best[key] = QuerySpec(
                    query=s.query.strip(),
                    query_type=s.query_type,
                    importance=clamp01(float(s.importance)),
                    reason=s.reason or "",
                )

        out = sorted(best.values(), key=lambda s: s.importance, reverse=True)
        if max_queries > 0:
            out = out[:max_queries]
        return out

    # ------------------------------------------------------------------
    # Query cleaning helpers
    # ------------------------------------------------------------------

    # Words too common / vague to be useful grep patterns on their own.
    _STOPWORDS: set[str] = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "and",
        "but",
        "or",
        "if",
        "while",
        "because",
        "until",
        "although",
        "also",
        "any",
        "like",
        "using",
        "used",
        "use",
        "show",
        "list",
        "find",
        "get",
        "set",
        "see",
        "need",
        "look",
        "make",
        "want",
        "know",
        "try",
        "take",
        "give",
        # Domain vague words:
        "code",
        "file",
        "function",
        "class",
        "method",
        "implement",
        "implementation",
        "relevant",
        "related",
        "information",
        "details",
        "specific",
        "actual",
        "content",
        "context",
        "available",
        "expose",
        "provide",
        "define",
        "section",
        "part",
        "include",
        "contain",
        "handle",
        "handles",
        "handling",
        "request",
        "requests",
        "response",
        "responses",
        "called",
        "calls",
        "call",
        "logic",
        "system",
        "module",
        "service",
        "data",
        "type",
        "types",
        "value",
        "values",
        "based",
        "support",
        "supported",
        "work",
        "works",
        "working",
        "source",
        "output",
        "input",
        "result",
        "results",
        "return",
        "returns",
        "returned",
        "check",
        "checks",
        "process",
        "processes",
        "anchor",
        "anchors",
        "citation",
        "citations",
    }

    # Identifiers: camelCase, PascalCase, snake_case, SCREAMING_CASE, dotted.paths
    _IDENT_RE = re.compile(
        r"\b("
        r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # PascalCase  (e.g. ResourceGovernor)
        r"|[a-z]+(?:_[a-z0-9]+)+"  # snake_case  (e.g. execute_spec)
        r"|[a-z]+[A-Z][a-zA-Z0-9]*"  # camelCase   (e.g. canLoadModel)
        r"|[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+"  # SCREAMING   (e.g. MCP_SERVER)
        r"|[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)+"  # dotted      (e.g. dcs.client)
        r")\b"
    )

    # File path fragments (e.g. src/mcp/mcp_server.cpp)
    # Requires either a file extension (.cpp, .py, etc.) or 2+ directory separators
    # to avoid false positives like "citations/anchors".
    _FILE_RE = re.compile(
        r"(?:^|[\s,;(])"
        r"((?:[A-Za-z_][\w.\-]*/)+"  # at least one dir separator
        r"[A-Za-z_][\w.\-]*\.[A-Za-z]{1,10}"  # filename with extension
        r"|(?:[A-Za-z_][\w.\-]*/){2,}"  # or 2+ dir separators
        r"[A-Za-z_][\w.\-]*)"  # filename (no ext required)
    )

    # Type prefix pattern: "semantic:", "grep:", "keyword:", etc.
    _TYPE_PREFIX_RE = re.compile(
        r"^\s*(?:semantic|grep|keyword|graph|get|list)\s*:\s*",
        re.IGNORECASE,
    )

    def _clean_query_for_grep(self, raw: str) -> str:
        """Transform a natural-language or prefixed query into a usable grep pattern.

        Returns an empty string if the input is too vague to produce a useful pattern.
        """
        raw = (raw or "").strip()
        if not raw:
            return ""

        # 1. Strip type prefixes ("semantic: ...", "grep: ...")
        cleaned = self._TYPE_PREFIX_RE.sub("", raw).strip()
        if not cleaned:
            return ""

        # 2. Strip wrapping quotes
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
            cleaned = cleaned[1:-1].strip()

        # 3. Extract file paths — if found, use the most specific one as the grep
        file_paths = self._FILE_RE.findall(cleaned)
        if file_paths:
            # Use the longest path (most specific)
            best_path = max(file_paths, key=len)
            return best_path[:80]

        # 4. Extract code identifiers (camelCase, snake_case, PascalCase, etc.)
        identifiers = self._IDENT_RE.findall(cleaned)
        if identifiers:
            # Deduplicate preserving order, take top 3
            seen: set[str] = set()
            unique: list[str] = []
            for ident in identifiers:
                if ident not in seen and len(ident) >= 3:
                    seen.add(ident)
                    unique.append(ident)
            if unique:
                # If just one, use it directly; if multiple, join with |
                pattern = "|".join(unique[:3])
                return pattern[:80]

        # 5. Fallback: extract non-stopword tokens ≥3 chars
        tokens = re.findall(r"[A-Za-z0-9_]+", cleaned)
        meaningful = [t for t in tokens if len(t) >= 3 and t.lower() not in self._STOPWORDS]

        if not meaningful:
            logger.debug("_clean_query_for_grep: no usable tokens from %r", raw)
            return ""

        # Take top 2-3 meaningful words, join as alternation
        terms = meaningful[:3]
        if len(terms) == 1:
            return terms[0][:80]

        pattern = "|".join(terms[:3])
        return pattern[:80]

    # Regex to find filenames like "mcp_server.cpp", "client.py", etc. in natural language
    _FILENAME_RE = re.compile(
        r"\b([a-zA-Z_][\w\-]*\.(?:cpp|h|hpp|py|ts|js|rs|go|java|c|cc|cxx|yaml|yml|toml|md|txt))\b"
    )

    def _extract_file_grep_specs(self, text: str) -> list[QuerySpec]:
        """Extract mentioned filenames from text and create high-priority GREP specs.

        When the critique says "Content of mcp_server.cpp", we should grep for
        the filename to find its path, rather than relying on generic patterns.
        """
        filenames = self._FILENAME_RE.findall(text)
        if not filenames:
            return []

        specs: list[QuerySpec] = []
        seen: set[str] = set()
        for fname in filenames:
            if fname in seen:
                continue
            seen.add(fname)
            # Use the filename stem (without extension) as grep pattern
            # since that matches both #include references and file content
            stem = fname.rsplit(".", 1)[0]
            specs.append(
                QuerySpec(
                    query=stem,
                    query_type=QueryType.GREP,
                    importance=0.85,  # higher than generic missing_info
                    reason=f"file '{fname}' mentioned in critique",
                )
            )
            logger.debug("_extract_file_grep_specs: %r -> grep for %r", fname, stem)
        return specs

    def _extract_task_grep_specs(self, task: str) -> list[QuerySpec]:
        """Extract code-relevant grep patterns directly from the task description.

        This catches identifiers and domain-specific patterns that the critique
        text may not mention explicitly but that are needed to find the right
        code.  E.g., "tools registered in mcp_server.cpp" should produce a
        grep for `registerTool` since that's the likely C++ function name.
        """
        specs: list[QuerySpec] = []
        seen: set[str] = set()

        # 1. Extract code identifiers from the task itself
        identifiers = self._IDENT_RE.findall(task)
        for ident in identifiers:
            if ident in seen or len(ident) < 4:
                continue
            seen.add(ident)
            specs.append(
                QuerySpec(
                    query=ident,
                    query_type=QueryType.GREP,
                    importance=0.75,
                    reason=f"identifier '{ident}' from task text",
                )
            )

        # 2. Infer compound camelCase/PascalCase patterns from verb+noun combos
        #    e.g., "tools registered" -> registerTool, "handle request" -> handleRequest
        _VERBS = (
            r"register(?:ed|s)?|handle[ds]?|create[ds]?|build[ds]?|parse[ds]?|"
            r"add(?:ed|s)?|init(?:ialize[ds]?)?|setup|start(?:ed|s)?"
        )
        _NOUNS = (
            r"tool|request|response|server|client|handler|config|session|"
            r"resource|prompt|document|query|command|method|route|event"
        )
        # Match verb...noun (e.g. "register the tool") or noun...verb (e.g. "tool names registered")
        _VN_RE = re.compile(rf"\b({_VERBS})\b.*?\b({_NOUNS})s?\b", re.IGNORECASE)
        _NV_RE = re.compile(rf"\b({_NOUNS})s?\b.*?\b({_VERBS})\b", re.IGNORECASE)

        def _normalize_verb(raw: str) -> str:
            v = raw.lower().rstrip("eds").rstrip("e")
            if v.endswith("ializ"):
                return "initialize"
            return v

        for m in _VN_RE.finditer(task):
            verb = _normalize_verb(m.group(1))
            noun = m.group(2).lower()
            compound = verb + noun.capitalize()
            if compound in seen or len(compound) < 6:
                continue
            seen.add(compound)
            specs.append(
                QuerySpec(
                    query=compound,
                    query_type=QueryType.GREP,
                    importance=0.90,
                    reason=f"inferred symbol '{compound}' from task verb+noun",
                )
            )
            logger.debug("_extract_task_grep_specs: inferred %r (verb+noun)", compound)

        for m in _NV_RE.finditer(task):
            noun = m.group(1).lower()
            verb = _normalize_verb(m.group(2))
            compound = verb + noun.capitalize()
            if compound in seen or len(compound) < 6:
                continue
            seen.add(compound)
            specs.append(
                QuerySpec(
                    query=compound,
                    query_type=QueryType.GREP,
                    importance=0.90,
                    reason=f"inferred symbol '{compound}' from task noun+verb",
                )
            )
            logger.debug("_extract_task_grep_specs: inferred %r (noun+verb)", compound)

        return specs

    def _fallback_decompose(self, task: str, max_queries: int) -> list[QuerySpec]:
        specs: list[QuerySpec] = []
        seen: set[tuple[str, str]] = set()

        # 1) Paths -> GET
        for m in _PATH_RE.finditer(task):
            path = m.group("path").strip().strip("'\"")
            if not path or "/" not in path:
                continue
            spec = QuerySpec(
                query=path,
                query_type=QueryType.GET,
                importance=0.95,
                reason="path mentioned in task",
            )
            key = spec_key(spec)
            if key in seen:
                continue
            seen.add(key)
            specs.append(spec)

        # 2) Error-like signals -> GREP
        if _ERROR_RE.search(task):
            # Take some quoted fragments / lines.
            fragments = set(re.findall(r"`([^`]{3,200})`", task))
            fragments |= set(re.findall(r"\"([^\"]{3,200})\"", task))
            fragments |= set(re.findall(r"'([^']{3,200})'", task))
            for frag in list(fragments)[:3]:
                spec = QuerySpec(
                    query=frag.strip(),
                    query_type=QueryType.GREP,
                    importance=0.8,
                    reason="error/pattern mentioned in task",
                )
                key = spec_key(spec)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(spec)

        # 3) Function/class names -> SEMANTIC
        symbols: set[str] = set()
        for m in _PY_SYMBOL_RE.finditer(task):
            sym = m.group(1)
            if not sym or len(sym) < 3:
                continue
            if sym.lower() in {"task", "file", "class", "def", "error"}:
                continue
            symbols.add(sym)
        for sym in list(sorted(symbols))[:3]:
            spec = QuerySpec(
                query=sym,
                query_type=QueryType.SEMANTIC,
                importance=0.7,
                reason="symbol mentioned in task",
            )
            key = spec_key(spec)
            if key in seen:
                continue
            seen.add(key)
            specs.append(spec)

        # 4) General task -> SEMANTIC
        spec = QuerySpec(
            query=task[:2000],
            query_type=QueryType.SEMANTIC,
            importance=0.6,
            reason="general task description",
        )
        key = spec_key(spec)
        if key not in seen:
            specs.append(spec)

        # Sort and trim.
        specs.sort(key=lambda s: s.importance, reverse=True)
        if max_queries > 0:
            specs = specs[:max_queries]
        logger.debug("fallback specs: %s", specs)
        return specs
