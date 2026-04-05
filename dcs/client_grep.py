from __future__ import annotations

import fnmatch
import hashlib
import re as _re
from pathlib import Path
from typing import Any

from dcs.client_parsing import get_str
from dcs.types import YAMSChunk


def parse_grep_file_paths(data: Any) -> list[tuple[str, int]]:
    if not isinstance(data, dict):
        return []

    structured = structured_grep_matches(data)
    if structured:
        entries: list[tuple[str, int]] = []
        for path, matches in structured.items():
            first = matches[0] if matches else {}
            raw_count = first.get("file_matches") if isinstance(first, dict) else None
            try:
                count = int(raw_count) if raw_count is not None else len(matches)
            except Exception:
                count = len(matches)
            entries.append((path, max(1, count)))
        entries.sort(key=lambda entry: entry[1], reverse=True)
        return entries

    output = data.get("output")
    if not isinstance(output, str) or not output.strip():
        return []

    header_re = _re.compile(r"^(/[^\s(]+)\s*\((\d+)\s+match", _re.MULTILINE)
    seen: set[str] = set()
    entries: list[tuple[str, int]] = []
    skip_exts = {".json", ".lock", ".log", ".bin", ".dat"}

    for match in header_re.finditer(output):
        path = match.group(1)
        count = int(match.group(2))
        if path in seen:
            continue
        if Path(path).suffix.lower() in skip_exts:
            continue
        seen.add(path)
        entries.append((path, count))

    if not entries:
        for line in output.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            path = line.split(":", 1)[0]
            if not path or path in seen:
                continue
            if not (path.startswith("/") or path.startswith(".")):
                continue
            if Path(path).suffix.lower() in skip_exts:
                continue
            seen.add(path)
            entries.append((path, 1))

    entries.sort(key=lambda entry: entry[1], reverse=True)
    return entries


def source_matches_filters(
    source: str,
    *,
    cwd: str | None = None,
    path_hint: str | None = None,
    include_hints: list[str] | None = None,
    exclude_hints: list[str] | None = None,
) -> bool:
    src = (source or "").strip()
    if not src:
        return False

    src_l = src.lower()
    cwd_norm = str(cwd or "").strip()
    if cwd_norm:
        try:
            src_abs = str(Path(src).resolve())
            cwd_abs = str(Path(cwd_norm).resolve())
            if not src_abs.startswith(cwd_abs):
                return False
        except Exception:
            if cwd_norm not in src:
                return False

    if path_hint:
        hint = path_hint.strip().strip("\"'")
        if hint:
            hint_l = hint.lower()
            if "/" in hint or "\\" in hint or "." in hint:
                if hint_l not in src_l:
                    return False
            elif Path(src).name.lower() != hint_l:
                return False

    for hint in include_hints or []:
        raw = (hint or "").strip()
        if not raw:
            continue
        raw_l = raw.lower()
        if any(ch in raw for ch in "*?[]"):
            if not fnmatch.fnmatch(src_l, raw_l):
                return False
        elif raw_l not in src_l:
            return False

    for hint in exclude_hints or []:
        raw = (hint or "").strip()
        if not raw:
            continue
        raw_l = raw.lower()
        if any(ch in raw for ch in "*?[]"):
            if fnmatch.fnmatch(src_l, raw_l):
                return False
        elif raw_l in src_l:
            return False

    return True


def structured_grep_matches(data: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(data, dict):
        return {}
    raw = data.get("matches")
    if not isinstance(raw, list):
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = get_str(item, "file")
        if not path:
            continue
        grouped.setdefault(path, []).append(item)
    return grouped


def read_file_context(
    filepath: str,
    pattern: str,
    *,
    context_lines: int = 10,
    max_matches: int = 10,
    max_chars: int = 4000,
) -> str | None:
    fpath = Path(filepath)
    if not fpath.is_file():
        return None
    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None

    lines = text.splitlines()
    if not lines:
        return None

    try:
        pat = _re.compile(pattern, _re.IGNORECASE)
    except _re.error:
        pat = None

    match_line_nums: list[int] = []
    for index, line in enumerate(lines):
        if pat is not None:
            if pat.search(line):
                match_line_nums.append(index)
        elif pattern.lower() in line.lower():
            match_line_nums.append(index)
        if len(match_line_nums) >= max_matches:
            break

    if not match_line_nums:
        return None

    regions: list[tuple[int, int]] = []
    for line_no in match_line_nums:
        start = max(0, line_no - context_lines)
        end = min(len(lines), line_no + context_lines + 1)
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    parts: list[str] = []
    total_chars = 0
    for start, end in regions:
        region_lines: list[str] = []
        for index in range(start, end):
            region_lines.append(f"{index + 1:>5}: {lines[index]}")
        region_text = "\n".join(region_lines)
        total_chars += len(region_text)
        if total_chars > max_chars:
            remaining = max_chars - (total_chars - len(region_text))
            if remaining > 100:
                parts.append(region_text[:remaining] + "\n  ... [truncated]")
            break
        parts.append(region_text)

    if not parts:
        return None
    return "\n  ---\n".join(parts)


def enrich_grep_results(
    data: Any,
    pattern: str,
    *,
    cwd: str | None = None,
    path_hint: str | None = None,
    include_hints: list[str] | None = None,
    exclude_hints: list[str] | None = None,
    max_files: int = 12,
    context_lines: int = 10,
    max_chars_per_file: int = 4000,
) -> list[YAMSChunk]:
    structured = structured_grep_matches(data)
    if structured:
        return chunks_from_structured_grep_matches(
            structured,
            pattern,
            cwd=cwd,
            path_hint=path_hint,
            include_hints=include_hints,
            exclude_hints=exclude_hints,
            max_files=max_files,
            context_lines=context_lines,
            max_chars_per_file=max_chars_per_file,
        )

    file_entries = parse_grep_file_paths(data)
    if not file_entries:
        return []

    if path_hint:
        hint = path_hint.strip()
        if hint:
            if "/" in hint:
                filtered = [entry for entry in file_entries if hint in entry[0]]
            else:
                filtered = [entry for entry in file_entries if Path(entry[0]).name == hint]
            if filtered:
                file_entries = filtered

    file_entries = [
        entry
        for entry in file_entries
        if source_matches_filters(
            entry[0],
            cwd=cwd,
            path_hint=path_hint,
            include_hints=include_hints,
            exclude_hints=exclude_hints,
        )
    ]

    match_count = data.get("match_count", 0) if isinstance(data, dict) else 0
    file_count = data.get("file_count", 0) if isinstance(data, dict) else 0
    pattern_tag = hashlib.md5((pattern or "").encode("utf-8")).hexdigest()[:8]

    chunks: list[YAMSChunk] = []
    pat_lower = (pattern or "").lower()
    tuned_context_lines = context_lines
    tuned_max_chars = max_chars_per_file
    tuned_max_matches = 10
    if "registertool" in pat_lower:
        tuned_context_lines = 1
        tuned_max_chars = 8000
        tuned_max_matches = 24

    for index, (fpath, file_matches) in enumerate(file_entries[:max_files]):
        context = read_file_context(
            fpath,
            pattern,
            context_lines=tuned_context_lines,
            max_matches=tuned_max_matches,
            max_chars=tuned_max_chars,
        )
        base_score = (
            min(1.0, 0.3 + (file_matches / max(1, match_count)) * 0.7) if match_count else 1.0
        )
        if context:
            header = f"# {fpath} ({file_matches} matches)\n"
            content = header + context
            chunks.append(
                YAMSChunk(
                    chunk_id=f"grep:{index}:{Path(fpath).name}:{pattern_tag}",
                    content=content,
                    score=base_score,
                    source=fpath,
                    metadata={
                        "match_count": match_count,
                        "file_count": file_count,
                        "file_matches": file_matches,
                        "enriched": True,
                    },
                )
            )
        else:
            chunks.append(
                YAMSChunk(
                    chunk_id=f"grep:{index}:{Path(fpath).name}:{pattern_tag}",
                    content=f"[file: {fpath}]",
                    score=0.3,
                    source=fpath,
                    metadata={
                        "match_count": match_count,
                        "file_count": file_count,
                        "enriched": False,
                    },
                )
            )

    return chunks


def chunks_from_structured_grep_matches(
    grouped_matches: dict[str, list[dict[str, Any]]],
    pattern: str,
    *,
    cwd: str | None = None,
    path_hint: str | None = None,
    include_hints: list[str] | None = None,
    exclude_hints: list[str] | None = None,
    max_files: int = 12,
    context_lines: int = 10,
    max_chars_per_file: int = 4000,
) -> list[YAMSChunk]:
    if not grouped_matches:
        return []

    entries: list[tuple[str, int]] = []
    for path, matches in grouped_matches.items():
        first = matches[0] if matches else {}
        raw_count = first.get("file_matches") if isinstance(first, dict) else None
        try:
            count = int(raw_count) if raw_count is not None else len(matches)
        except Exception:
            count = len(matches)
        entries.append((path, max(1, count)))

    if path_hint:
        hint = path_hint.strip()
        if hint:
            if "/" in hint:
                entries = [entry for entry in entries if hint in entry[0]]
            else:
                entries = [entry for entry in entries if Path(entry[0]).name == hint]

    entries = [
        entry
        for entry in entries
        if source_matches_filters(
            entry[0],
            cwd=cwd,
            path_hint=path_hint,
            include_hints=include_hints,
            exclude_hints=exclude_hints,
        )
    ]

    entries.sort(key=lambda entry: entry[1], reverse=True)
    if not entries:
        return []

    pattern_tag = hashlib.md5((pattern or "").encode("utf-8")).hexdigest()[:8]
    chunks: list[YAMSChunk] = []
    pat_terms = [
        term.lower() for term in _re.findall(r"[A-Za-z0-9_]+", pattern or "") if len(term) >= 3
    ]

    for index, (fpath, file_matches) in enumerate(entries[:max_files]):
        matches = grouped_matches.get(fpath, [])
        rendered: list[str] = []
        total_chars = 0
        has_structured_context = False

        for match in matches:
            line_text = get_str(match, "line_text")
            before = (
                match.get("context_before") if isinstance(match.get("context_before"), list) else []
            )
            after = (
                match.get("context_after") if isinstance(match.get("context_after"), list) else []
            )
            try:
                line_no = int(match.get("line_number") or 0)
            except Exception:
                line_no = 0
            if line_text.strip() or before or after or line_no > 0:
                has_structured_context = True
                break

        if has_structured_context:
            for match in matches:
                try:
                    line_no = int(match.get("line_number") or 0)
                except Exception:
                    line_no = 0
                line_text = get_str(match, "line_text")
                before = (
                    match.get("context_before")
                    if isinstance(match.get("context_before"), list)
                    else []
                )
                after = (
                    match.get("context_after")
                    if isinstance(match.get("context_after"), list)
                    else []
                )
                if not (line_text.strip() or before or after or line_no > 0):
                    continue
                if before:
                    for item in before[-context_lines:]:
                        rendered.append(f"      {str(item)}")
                if line_no > 0:
                    rendered.append(f"{line_no:>5}: {line_text}")
                else:
                    rendered.append(f"      {line_text}")
                if after:
                    for item in after[:context_lines]:
                        rendered.append(f"      {str(item)}")
                rendered.append("  ---")
                total_chars = sum(len(item) + 1 for item in rendered)
                if total_chars >= max_chars_per_file:
                    break
        else:
            fallback_ctx = read_file_context(
                fpath,
                pattern,
                context_lines=context_lines,
                max_matches=20,
                max_chars=max_chars_per_file,
            )
            if fallback_ctx:
                rendered = fallback_ctx.splitlines()

        if rendered and rendered[-1] == "  ---":
            rendered.pop()

        body = "\n".join(rendered)
        if len(body) > max_chars_per_file:
            body = body[:max_chars_per_file] + "\n  ... [truncated]"

        term_hits = 0
        term_total = max(1, len(pat_terms))
        if pat_terms:
            lower_body = body.lower()
            term_hits = sum(1 for term in pat_terms if term in lower_body)
        completeness = min(1.0, (term_hits / term_total) if pat_terms else 0.6)
        score = min(1.0, 0.25 + min(0.55, file_matches / 20.0) + (0.20 * completeness))

        header = f"# {fpath} ({file_matches} matches)\n"
        content = header + body if body else header
        chunks.append(
            YAMSChunk(
                chunk_id=f"grep:{index}:{Path(fpath).name}:{pattern_tag}",
                content=content,
                score=score,
                source=fpath,
                metadata={
                    "file_matches": file_matches,
                    "term_hits": term_hits,
                    "term_total": term_total,
                    "enriched": True,
                    "structured": True,
                },
            )
        )

    return chunks
