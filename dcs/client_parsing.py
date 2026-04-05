from __future__ import annotations

import json
import re as _re
from pathlib import Path
from typing import Any

from dcs.shared import is_noise_source
from dcs.types import YAMSChunk


def is_code_source(path: str) -> bool:
    p = (path or "").lower()
    if not p:
        return False
    return p.endswith(
        (
            ".cpp",
            ".c",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".py",
            ".rs",
            ".ts",
            ".tsx",
            ".js",
            ".go",
            ".java",
        )
    )


def maybe_path_like(text: str) -> str | None:
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return None
    if "\n" in raw:
        raw = raw.splitlines()[0].strip().strip("\"'")
    if not (
        "/" in raw
        or raw.endswith(
            (
                ".cpp",
                ".cc",
                ".cxx",
                ".c",
                ".h",
                ".hpp",
                ".py",
                ".rs",
                ".ts",
                ".js",
                ".go",
                ".java",
            )
        )
    ):
        return None
    return raw


def normalize_search_source(path: str, snippet: str, title: str = "") -> str:
    src = (path or "").strip()
    if src and not is_noise_source(src):
        return src

    for candidate in (snippet, title):
        maybe = maybe_path_like(candidate)
        if maybe and not is_noise_source(maybe):
            return maybe
    return src


def get_str(mapping: dict[str, Any], key: str) -> str:
    val = mapping.get(key)
    return val if isinstance(val, str) else ""


def extract_tool_data(tool_result: dict[str, Any]) -> Any:
    if not isinstance(tool_result, dict):
        return tool_result

    sc = tool_result.get("structuredContent")
    if isinstance(sc, dict) and sc:
        if any(k in sc for k in ("results", "matches", "documents", "paths", "ready", "steps")):
            return sc
    if isinstance(sc, dict) and isinstance(sc.get("data"), (dict, list, str, int, float, bool)):
        return sc.get("data")

    content = tool_result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            text = first["text"]
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    return tool_result


def query_terms(query: str) -> list[str]:
    out: list[str] = []
    for token in _re.findall(r"[A-Za-z0-9_./-]+", query or ""):
        raw = token.strip("._/-")
        lower = raw.lower()
        if len(lower) < 3:
            continue
        if lower in {
            "what",
            "does",
            "used",
            "use",
            "with",
            "from",
            "that",
            "this",
            "how",
            "list",
            "each",
            "focus",
            "explain",
            "describe",
            "work",
            "works",
        }:
            continue
        if lower not in out:
            out.append(lower)
        camel_parts = _re.findall(r"[A-Z]?[a-z]+|[0-9]+", raw)
        for part in camel_parts:
            part_l = part.lower()
            if len(part_l) >= 3 and part_l not in out:
                out.append(part_l)
        for part in _re.findall(r"[a-z]+|[0-9]+", lower.replace("_", "-")):
            if len(part) >= 3 and part not in out:
                out.append(part)
    return out


def identifier_terms(text: str) -> list[str]:
    out: list[str] = []
    for token in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        lower = token.lower()
        if len(lower) < 3:
            continue
        if lower not in out:
            out.append(lower)
        camel_parts = _re.findall(r"[A-Z]?[a-z]+|[0-9]+", token)
        for part in camel_parts:
            part_l = part.lower()
            if len(part_l) >= 3 and part_l not in out:
                out.append(part_l)
    return out


def code_relevance_score(query: str, chunk: YAMSChunk) -> float:
    source = (chunk.source or "").strip()
    source_l = source.lower()
    content = chunk.content or ""
    content_l = content.lower()
    q_terms = query_terms(query)
    if not q_terms:
        return float(chunk.score or 0.0)

    score = float(chunk.score or 0.0)
    if source:
        if is_code_source(source):
            score += 0.18
        if is_noise_source(source):
            score -= 0.35
        if "/src/" in source_l or "/include/" in source_l:
            score += 0.08
        if "/external/agent/results/" in source_l or "/eval/tasks/" in source_l:
            score -= 0.40

    path_terms = set(identifier_terms(source))
    content_terms = set(identifier_terms(content))
    direct_hits = 0
    identifier_hits = 0
    partial_hits = 0
    basename = Path(source).name.lower() if source else ""

    for term in q_terms[:16]:
        if source_l and term in source_l:
            direct_hits += 1
        if term in path_terms or term in content_terms:
            identifier_hits += 1
        elif basename and term in basename:
            partial_hits += 1

    score += min(0.30, 0.08 * direct_hits)
    score += min(0.28, 0.07 * identifier_hits)
    score += min(0.08, 0.04 * partial_hits)

    query_file = None
    match = _re.search(r"([A-Za-z_][\w\-]*\.[A-Za-z0-9]{1,8})", (query or "").lower())
    if match:
        query_file = match.group(1)
    if query_file and query_file in basename:
        score += 0.25

    if "# " in content[:4] and "matches" in content[:80]:
        score += 0.03
    if "line=" in content_l or "char=" in content_l:
        score += 0.03

    return max(0.0, min(1.0, score))


def rerank_code_chunks(
    query: str, chunks: list[YAMSChunk], *, limit: int | None = None
) -> list[YAMSChunk]:
    rescored: list[tuple[float, int, YAMSChunk]] = []
    for idx, chunk in enumerate(chunks):
        rescored.append((code_relevance_score(query, chunk), idx, chunk))
    rescored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    ordered = []
    for score, _, chunk in rescored:
        chunk.score = score
        ordered.append(chunk)
    if limit is not None:
        return ordered[: max(0, int(limit))]
    return ordered
