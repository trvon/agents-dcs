"""Graph-query codemap builder for the DCS pipeline.

Queries the YAMS knowledge graph to build a structural map of the codebase,
providing the agent with "spatial awareness" of files, functions, classes,
and their relationships. The codemap is injected as a persistent context
prefix before task-specific retrieved chunks.

Design:
  1. Query node type counts to understand KG contents
  2. For a given task, identify relevant directories/files
  3. Traverse contains/calls/includes edges to build structure
  4. Render a compact tree within a token budget
  5. Return as a ContextBlock for injection into the assembler

Key YAMS graph relations used:
  - contains: directory→file, file→symbol, class→method
  - calls: function→function call graph
  - includes: file import/include relationships
  - defined_in: symbol→document
  - inherits/implements: class hierarchy
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .types import ContextBlock

logger = logging.getLogger(__name__)

_CODEMAP_NOISE_RE = re.compile(
    r"(?:^|/)(?:external/agent/results|external/agent/eval/tasks|\.git|build|dist|__pycache__|node_modules)(?:/|$)"
)


@dataclass
class GraphNode:
    """A node in the structural codemap."""

    node_key: str
    node_type: str  # file, function, class, directory, etc.
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    children: list[GraphNode] = field(default_factory=list)
    depth: int = 0


@dataclass
class CodemapResult:
    """Result of building a codemap."""

    tree_text: str  # Rendered tree (compact, for injection)
    context_block: ContextBlock  # Ready for assembler injection
    node_count: int = 0
    edge_count: int = 0
    query_count: int = 0
    latency_ms: float = 0.0
    node_type_counts: dict[str, int] = field(default_factory=dict)


class CodemapBuilder:
    """Builds a structural codemap from the YAMS knowledge graph.

    Usage:
        async with YAMSClient() as client:
            builder = CodemapBuilder(client, token_budget=512)
            result = await builder.build(task="implement caching for search")
    """

    def __init__(
        self,
        yams_client: Any,  # YAMSClient (avoid circular import)
        token_budget: int = 512,
        max_files: int = 5,
        max_symbols_per_file: int = 8,
        max_depth: int = 2,
        include_type_counts: bool = False,
    ):
        self._client = yams_client
        self._token_budget = token_budget
        self._max_files = max_files
        self._max_symbols_per_file = max_symbols_per_file
        self._max_depth = min(max_depth, 5)  # YAMS caps at 5
        self._query_count = 0
        self._include_type_counts = bool(include_type_counts)

    async def build(
        self,
        task: str = "",
        focus_paths: list[str] | None = None,
        focus_types: list[str] | None = None,
    ) -> CodemapResult:
        """Build a codemap, optionally focused on specific paths or node types.

        Args:
            task: The task description (used to focus on relevant areas).
            focus_paths: Specific file/directory paths to emphasize.
            focus_types: Node types to include (default: file, function, class).

        Returns:
            CodemapResult with rendered tree and context block.
        """
        t0 = time.perf_counter()
        self._query_count = 0

        focus_types = focus_types or ["file", "function", "class"]

        # Prefer task-anchored file selection over broad file listings.
        if not focus_paths and task:
            focus_paths = await self._select_focus_paths(task)

        # Step 1: Discover what's in the KG
        type_counts = await self._get_type_counts() if self._include_type_counts else {}
        logger.info("codemap: KG node types: %s", type_counts)

        # Step 2: Get file nodes (either focused or top-level)
        file_nodes = await self._get_file_nodes(focus_paths, task)

        # Step 3: For each file, get contained symbols
        total_edges = 0
        for fnode in file_nodes[: self._max_files]:
            symbols, edges = await self._get_file_symbols(fnode)
            fnode.children = symbols[: self._max_symbols_per_file]
            total_edges += edges

        # Step 4: Render tree
        tree_text = self._render_tree(file_nodes, type_counts)

        # Step 5: Truncate to budget
        tree_text = self._truncate_to_budget(tree_text)

        # Step 6: Build ContextBlock
        token_count = self._estimate_tokens(tree_text)
        context_block = ContextBlock(
            content=tree_text,
            sources=["yams-knowledge-graph"],
            chunk_ids=["codemap"],
            token_count=token_count,
            budget=self._token_budget,
            utilization=min(1.0, token_count / max(1, self._token_budget)),
            chunks_included=1,
            chunks_considered=1,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        node_count = sum(1 + len(f.children) for f in file_nodes)

        result = CodemapResult(
            tree_text=tree_text,
            context_block=context_block,
            node_count=node_count,
            edge_count=total_edges,
            query_count=self._query_count,
            latency_ms=latency_ms,
            node_type_counts=type_counts,
        )

        logger.info(
            "codemap: built in %.0fms — %d nodes, %d edges, %d queries, %d tokens",
            latency_ms,
            node_count,
            total_edges,
            self._query_count,
            token_count,
        )
        return result

    async def _select_focus_paths(self, task: str) -> list[str]:
        try:
            chunks = await self._client.search(task, limit=max(self._max_files * 2, 12))
        except Exception as e:
            logger.warning("codemap: focus path search failed: %s", e)
            return []

        seen: set[str] = set()
        out: list[str] = []
        for chunk in chunks:
            source = (chunk.source or "").strip()
            if not source or source in seen or not self._is_code_path(source):
                continue
            if not self._path_matches_task(source, task) and len(out) >= max(
                2, self._max_files // 2
            ):
                continue
            seen.add(source)
            out.append(source)
            if len(out) >= self._max_files:
                break
        return out

    @staticmethod
    def _task_terms(task: str) -> list[str]:
        out: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_./-]+", task or ""):
            raw = token.strip("._/-")
            t = raw.lower()
            if len(t) < 4:
                continue
            if t in {"what", "does", "used", "use", "with", "from", "that", "this", "how"}:
                continue
            if t not in out:
                out.append(t)
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+", raw)
            for part in camel_parts:
                part_l = part.lower()
                if len(part_l) >= 4 and part_l not in out:
                    out.append(part_l)
            for part in re.findall(r"[a-z]+|[0-9]+", t.replace("_", "-")):
                if len(part) >= 4 and part not in out:
                    out.append(part)
        return out

    def _path_matches_task(self, path: str, task: str) -> bool:
        p = (path or "").lower()
        terms = self._task_terms(task)
        return any(term in p for term in terms[:12])

    @staticmethod
    def _is_code_symbol_type(node_type: str) -> bool:
        normalized = str(node_type or "").lower()
        return normalized in {
            "function",
            "function_version",
            "class",
            "class_version",
            "method",
            "method_version",
            "struct",
            "struct_version",
            "enum",
            "enum_version",
            "interface",
            "interface_version",
            "trait",
            "trait_version",
            "namespace",
            "namespace_version",
            "variable",
            "variable_version",
            "constant",
            "constant_version",
            "typedef",
            "typedef_version",
            "macro",
            "macro_version",
            "field",
            "field_version",
        }

    @staticmethod
    def _is_code_path(path: str) -> bool:
        p = (path or "").strip()
        if not p or _CODEMAP_NOISE_RE.search(p):
            return False
        return p.endswith(
            (
                ".cpp",
                ".cc",
                ".cxx",
                ".c",
                ".hpp",
                ".h",
                ".hh",
                ".py",
                ".rs",
                ".ts",
                ".tsx",
                ".js",
                ".go",
                ".java",
            )
        )

    @staticmethod
    def _symbol_display(item: dict[str, Any]) -> str:
        label = str(item.get("label") or "").strip()
        node_key = str(item.get("nodeKey") or item.get("node_key") or "")
        if not label and node_key:
            label = node_key
        if "@" in label:
            label = label.split("@", 1)[0]
        if "::" in label:
            label = label.split("::")[-1]
        return label.strip()

    @staticmethod
    def _normalize_symbol_type(node_type: str, node_key: str) -> str:
        normalized = str(node_type or "").lower().replace("_version", "")
        if normalized in {"function", "class", "method", "struct", "enum", "namespace"}:
            return normalized
        key_type = str(node_key or "").split(":", 1)[0].lower()
        if key_type in {"function", "class", "method", "struct", "enum", "namespace"}:
            return key_type
        return normalized

    @staticmethod
    def _path_from_graph_item(item: dict[str, Any]) -> str:
        label = str(item.get("label") or "").strip()
        if label.startswith("/"):
            return label
        node_key = str(item.get("nodeKey") or item.get("node_key") or "")
        marker = "path:file:"
        if marker in node_key:
            return node_key.split(marker, 1)[1].split("@", 1)[0]
        return label

    async def _get_type_counts(self) -> dict[str, int]:
        """Query KG for node type counts."""
        try:
            self._query_count += 1
            data = await self._client.graph_query(list_types=True)
            counts = data.get("node_type_counts") or data.get("nodeTypeCounts")
            if isinstance(counts, dict):
                return {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.warning("codemap: list_types failed: %s", e)
        return {}

    async def _get_file_nodes(
        self,
        focus_paths: list[str] | None,
        task: str,
    ) -> list[GraphNode]:
        """Get file nodes, optionally focused on specific paths."""
        nodes: list[GraphNode] = []

        if focus_paths:
            # Query each focus path directly
            for path in focus_paths[:20]:
                node = await self._query_file_node(path)
                if node:
                    nodes.append(node)
            if nodes:
                return nodes

        # No focus paths: list file nodes from KG
        try:
            self._query_count += 1
            data = await self._client.graph_query(
                list_type="file",
                limit=self._max_files,
            )
            connected = data.get("connected_nodes") or data.get("connectedNodes") or []
            # list_type response may use different structure; also check origin
            if not connected:
                # For list_type queries, nodes may be in a flat array structure
                nodes_data = data.get("nodes") or data.get("results") or []
                if isinstance(nodes_data, list):
                    connected = nodes_data

            for item in connected:
                if not isinstance(item, dict):
                    continue
                node_key = str(item.get("nodeKey") or item.get("node_key") or "")
                node_type = str(item.get("type") or "file")
                label = str(item.get("label") or "")
                props = item.get("properties") or {}
                if isinstance(props, str):
                    props = {}

                # Extract short path from node_key (strip "file:" prefix)
                display = self._path_from_graph_item(item) or node_key
                if not self._is_code_path(display):
                    continue

                nodes.append(
                    GraphNode(
                        node_key=node_key,
                        node_type=node_type,
                        label=label or display,
                        properties=props if isinstance(props, dict) else {},
                    )
                )
        except Exception as e:
            logger.warning("codemap: list file nodes failed: %s", e)

        # If we got nothing from the graph, try using search to find relevant files
        if not nodes and task:
            nodes = await self._search_for_relevant_files(task)

        return nodes

    async def _search_for_relevant_files(self, task: str) -> list[GraphNode]:
        """Use YAMS search to find files relevant to the task, then wrap as graph nodes."""
        try:
            chunks = await self._client.search(task, limit=self._max_files)
            nodes: list[GraphNode] = []
            seen_sources: set[str] = set()
            for chunk in chunks:
                source = (chunk.source or "").strip()
                if not source or source in seen_sources or not self._is_code_path(source):
                    continue
                seen_sources.add(source)
                nodes.append(
                    GraphNode(
                        node_key=f"path:file:{source}",
                        node_type="file",
                        label=source,
                    )
                )
            return nodes
        except Exception as e:
            logger.warning("codemap: search fallback failed: %s", e)
            return []

    async def _query_file_node(self, path: str) -> GraphNode | None:
        """Query a specific file node by path."""
        # Try with file: prefix first
        node_key = path if path.startswith("file:") else f"file:{path}"
        try:
            self._query_count += 1
            data = await self._client.graph_query(node_key=node_key, depth=0)
            origin = data.get("origin")
            if isinstance(origin, dict) and origin.get("nodeKey"):
                return GraphNode(
                    node_key=str(origin.get("nodeKey", "")),
                    node_type=str(origin.get("type", "file")),
                    label=str(origin.get("label", path)),
                    properties=origin.get("properties") or {},
                )
        except Exception as e:
            logger.debug("codemap: file node query failed for %s: %s", path, e)

        # Fallback: try by name
        try:
            self._query_count += 1
            data = await self._client.graph_query(name=path, depth=0)
            origin = data.get("origin")
            if isinstance(origin, dict) and origin.get("nodeKey"):
                return GraphNode(
                    node_key=str(origin.get("nodeKey", "")),
                    node_type=str(origin.get("type", "file")),
                    label=str(origin.get("label", path)),
                    properties=origin.get("properties") or {},
                )
        except Exception as e:
            logger.debug("codemap: name query failed for %s: %s", path, e)

        return None

    async def _get_file_symbols(self, file_node: GraphNode) -> tuple[list[GraphNode], int]:
        """Get symbols (functions, classes) contained in a file."""
        try:
            self._query_count += 1
            data = await self._client.graph_query(
                node_key=file_node.node_key,
                relation="contains",
                depth=1,
                include_properties=True,
                limit=self._max_symbols_per_file * 2,  # fetch extra, trim later
            )

            connected = data.get("connected_nodes") or data.get("connectedNodes") or []
            edge_count = int(
                data.get("total_edges_traversed")
                or data.get("totalEdgesTraversed")
                or len(connected)
            )

            symbols: list[GraphNode] = []
            for item in connected:
                if not isinstance(item, dict):
                    continue
                node_key = str(item.get("nodeKey") or item.get("node_key") or "")
                node_type = self._normalize_symbol_type(item.get("type"), node_key)
                label = self._symbol_display(item)
                props = item.get("properties") or {}
                if isinstance(props, str):
                    props = {}

                # Filter to code symbols
                if not self._is_code_symbol_type(node_type):
                    continue

                # Extract short name from node_key
                display = label or node_key
                if "::" in display or "@" in display:
                    # Take the short name after last :: or before @
                    short = display.split("@")[0] if "@" in display else display
                    parts = short.split("::")
                    display = parts[-1] if parts else display

                symbols.append(
                    GraphNode(
                        node_key=node_key,
                        node_type=node_type,
                        label=display,
                        properties=props if isinstance(props, dict) else {},
                        depth=1,
                    )
                )

            # Sort: classes first, then functions, then others
            type_order = {
                "class": 0,
                "struct": 0,
                "interface": 0,
                "trait": 0,
                "enum": 1,
                "namespace": 1,
                "function": 2,
                "method": 2,
                "variable": 3,
                "constant": 3,
                "typedef": 3,
                "macro": 3,
            }
            symbols.sort(key=lambda s: (type_order.get(s.node_type, 9), s.label))
            return symbols, edge_count

        except Exception as e:
            logger.debug("codemap: get symbols failed for %s: %s", file_node.node_key, e)
            return [], 0

    def _render_tree(
        self,
        file_nodes: list[GraphNode],
        type_counts: dict[str, int],
    ) -> str:
        """Render the codemap as a compact, readable tree."""
        lines: list[str] = []

        # Header with KG stats
        if type_counts:
            stats_parts = []
            for t in ("file", "function", "class", "directory"):
                if t in type_counts:
                    stats_parts.append(f"{type_counts[t]} {t}s")
            if stats_parts:
                lines.append(f"# Codemap ({', '.join(stats_parts)})")
            else:
                lines.append("# Codemap")
        else:
            lines.append("# Codemap")

        lines.append("")

        if not file_nodes:
            lines.append("(no structural data available)")
            return "\n".join(lines)

        # Group files by directory
        dir_groups: dict[str, list[GraphNode]] = {}
        for fnode in file_nodes:
            path = fnode.label or fnode.node_key
            if path.startswith("file:"):
                path = path[5:]
            # Extract directory
            parts = path.rsplit("/", 1)
            dir_name = parts[0] if len(parts) > 1 else "."
            dir_groups.setdefault(dir_name, []).append(fnode)

        # Render each directory group
        for dir_name in sorted(dir_groups.keys()):
            files = dir_groups[dir_name]
            lines.append(f"## {dir_name}/")

            for fnode in sorted(files, key=lambda f: f.label):
                fname = fnode.label or fnode.node_key
                if fname.startswith("file:"):
                    fname = fname[5:]
                # Show just the filename, not full path
                short_name = fname.rsplit("/", 1)[-1] if "/" in fname else fname
                lines.append(f"  {short_name}")

                # Show contained symbols
                if fnode.children:
                    for sym in fnode.children:
                        type_tag = sym.node_type
                        normalized = type_tag.replace("_version", "")
                        if normalized in ("function", "method"):
                            prefix = "fn"
                        elif normalized in ("class", "struct"):
                            prefix = "cls"
                        elif normalized == "enum":
                            prefix = "enum"
                        elif normalized == "namespace":
                            prefix = "ns"
                        else:
                            prefix = normalized[:3]
                        lines.append(f"    [{prefix}] {sym.label}")

            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate rendered tree to fit within token budget."""
        tokens = self._estimate_tokens(text)
        if tokens <= self._token_budget:
            return text

        # Truncate line by line from the end, preserving header
        lines = text.split("\n")
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                header_end = i
                break

        # Keep header + as many lines as fit
        result_lines = lines[:header_end] if header_end > 0 else lines[:2]
        for line in lines[header_end:]:
            candidate = "\n".join(result_lines + [line])
            if self._estimate_tokens(candidate + "\n(truncated)\n") > self._token_budget:
                break
            result_lines.append(line)

        result_lines.append("(truncated)")
        return "\n".join(result_lines) + "\n"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for code."""
        return max(1, len(text) // 4)


__all__ = ["CodemapBuilder", "CodemapResult", "GraphNode"]
