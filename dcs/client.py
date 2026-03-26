"""Async YAMS MCP client (stdio JSON-RPC over NDJSON).

This client spawns `yams serve` as a subprocess and communicates with it using the
MCP stdio transport: newline-delimited JSON-RPC 2.0 messages.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import re as _re
import time
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult

logger = logging.getLogger(__name__)


def _is_noise_source(path: str) -> bool:
    p = (path or "").lower()
    if not p:
        return False
    if "/tests/" in p or "/docs/" in p or "/benchmarks/" in p:
        return True
    return p.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".lock"))


class YAMSClientError(RuntimeError):
    pass


class YAMSProtocolError(YAMSClientError):
    pass


class YAMSProcessError(YAMSClientError):
    pass


@dataclass
class _Pending:
    future: asyncio.Future[dict[str, Any]]
    method: str


class YAMSClient:
    """YAMS MCP client over stdio.

    Usage:
        async with YAMSClient() as client:
            chunks = await client.search("foo")
    """

    def __init__(
        self,
        *,
        yams_binary: str = "yams",
        data_dir: str | None = None,
        # Back-compat with callers that use PipelineConfig.yams_data_dir naming.
        yams_data_dir: str | None = None,
        # Scope search/grep results to files under this directory tree.
        cwd: str | None = None,
        # Best-effort: callers may provide fusion-style weights. We don't have a
        # first-class MCP field for these, but we can use them for heuristics.
        search_weights: dict[str, float] | None = None,
        request_timeout_s: float = 15.0,
        start_timeout_s: float = 10.0,
        stop_timeout_s: float = 5.0,
        protocol_version: str = "2025-06-18",
        client_name: str = "dcs-yams-client",
        client_version: str = "0.0.1",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._yams_binary = yams_binary
        self._data_dir = data_dir if data_dir is not None else yams_data_dir
        self._cwd = cwd
        self._request_timeout_s = request_timeout_s
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._protocol_version = protocol_version
        self._client_name = client_name
        self._client_version = client_version
        self._extra_env = dict(extra_env or {})

        self.search_weights: dict[str, float] = dict(search_weights or {})

        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None

        self._send_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}

        self._initialized = False
        self._stopping = False
        self._tool_names: set[str] | None = None

    @property
    def yams_binary(self) -> str:
        return self._yams_binary

    @yams_binary.setter
    def yams_binary(self, value: str) -> None:
        if self.is_running:
            raise YAMSClientError("Cannot change yams_binary while running")
        self._yams_binary = str(value)

    @property
    def yams_data_dir(self) -> str | None:
        return self._data_dir

    @yams_data_dir.setter
    def yams_data_dir(self, value: str | None) -> None:
        if self.is_running:
            raise YAMSClientError("Cannot change yams_data_dir while running")
        self._data_dir = None if value is None else str(value)

    async def __aenter__(self) -> YAMSClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if self._proc is not None:
            return

        env = os.environ.copy()
        # Keep stdout clean (protocol stream). Server logs to stderr.
        env.setdefault("YAMS_MCP_QUIET", "1")
        if self._data_dir is not None:
            env["YAMS_DATA_DIR"] = self._data_dir
        env.update(self._extra_env)

        logger.info("Starting yams MCP server: %s serve", self._yams_binary)
        self._proc = await asyncio.create_subprocess_exec(
            self._yams_binary,
            "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        self._stdout_task = asyncio.create_task(self._read_stdout_loop(), name="yams-mcp-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr_loop(), name="yams-mcp-stderr")
        self._wait_task = asyncio.create_task(self._wait_loop(), name="yams-mcp-wait")

        try:
            await asyncio.wait_for(self._initialize_handshake(), timeout=self._start_timeout_s)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return

        logger.info("Stopping yams MCP server")

        self._stopping = True

        # Best-effort graceful shutdown.
        try:
            if proc.returncode is None:
                try:
                    await self._request("shutdown", params={}, timeout_s=1.5)
                except Exception:
                    pass
                try:
                    await self._notify("exit", params={})
                except Exception:
                    pass

                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

                try:
                    await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout_s)
                except TimeoutError:
                    logger.warning("yams serve did not exit in time; terminating")
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout_s)
                    except TimeoutError:
                        logger.error("yams serve did not terminate; killing")
                        proc.kill()
                        await proc.wait()
        finally:
            self._proc = None
            self._initialized = False
            self._stopping = False

            for task in (self._stdout_task, self._stderr_task, self._wait_task):
                if task is not None and not task.done():
                    task.cancel()

            self._stdout_task = None
            self._stderr_task = None
            self._wait_task = None

            self._fail_all_pending(YAMSProcessError("YAMS MCP client stopped"))

    async def _initialize_handshake(self) -> None:
        init_params: dict[str, Any] = {
            "protocolVersion": self._protocol_version,
            "capabilities": {},
            "clientInfo": {"name": self._client_name, "version": self._client_version},
        }
        res = await self._request("initialize", params=init_params, timeout_s=self._start_timeout_s)
        logger.debug("MCP initialize result: %s", res)
        await self._notify("notifications/initialized", params={})
        self._initialized = True

        # Best-effort tool cache for feature detection (e.g. composite "query").
        try:
            await self.refresh_tools()
        except Exception:
            self._tool_names = None

    def _require_process(self) -> None:
        if self._proc is None:
            raise YAMSProcessError("YAMS server is not started")
        if self._proc.returncode is not None:
            raise YAMSProcessError(f"YAMS server exited with code {self._proc.returncode}")

    def _require_ready(self) -> None:
        self._require_process()
        if not self._initialized:
            raise YAMSProtocolError("MCP handshake not completed")

    async def _notify(self, method: str, *, params: dict[str, Any] | None = None) -> None:
        self._require_process()
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await self._send_message(msg)

    async def _request(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._require_process()
        timeout_s = self._request_timeout_s if timeout_s is None else timeout_s

        async with self._send_lock:
            req_id = self._next_id
            self._next_id += 1
            fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[req_id] = _Pending(future=fut, method=method)

            msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
            await self._send_message(msg)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except TimeoutError as e:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"Timed out waiting for JSON-RPC response: {method}") from e

    async def _send_message(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise YAMSProcessError("YAMS subprocess stdin is not available")
        if proc.returncode is not None:
            raise YAMSProcessError(f"YAMS subprocess exited with code {proc.returncode}")

        line = json.dumps(msg, ensure_ascii=True, separators=(",", ":")) + "\n"
        logger.debug("MCP send: %s", line.rstrip("\n"))
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()

    async def _read_stdout_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                line = await stream.readline()
                if not line:
                    # During intentional shutdown we expect stdout to close.
                    if self._stopping:
                        return
                    raise EOFError("YAMS MCP stdout closed")
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                logger.debug("MCP recv: %s", raw)
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON from yams stdout: %r", raw)
                    continue

                # Server can respond with batch arrays.
                if isinstance(msg, list):
                    for entry in msg:
                        if isinstance(entry, dict):
                            self._handle_incoming(entry)
                    continue

                if isinstance(msg, dict):
                    self._handle_incoming(msg)
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Avoid noisy errors for graceful stop.
            if self._stopping and isinstance(e, EOFError):
                return
            logger.error("YAMS MCP stdout loop ended: %s", e)
            self._fail_all_pending(YAMSProcessError(str(e)))

    @staticmethod
    def _looks_like_hash(s: str) -> bool:
        s = (s or "").strip()
        if not (8 <= len(s) <= 64):
            return False
        for c in s.lower():
            if c not in "0123456789abcdef":
                return False
        return True

    async def _read_stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        stream = self._proc.stderr
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    logger.debug("yams serve stderr: %s", text)
        except asyncio.CancelledError:
            return

    async def _wait_loop(self) -> None:
        assert self._proc is not None
        try:
            rc = await self._proc.wait()
            if rc != 0:
                logger.error("yams serve exited with code %s", rc)
            self._fail_all_pending(YAMSProcessError(f"YAMS server exited with code {rc}"))
        except asyncio.CancelledError:
            return

    def _handle_incoming(self, msg: dict[str, Any]) -> None:
        # Notifications
        if "id" not in msg:
            return

        req_id = msg.get("id")
        if not isinstance(req_id, int):
            return

        pending = self._pending.pop(req_id, None)
        if pending is None:
            return

        fut = pending.future
        if fut.done():
            return

        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            if isinstance(err, dict):
                code = err.get("code")
                message = err.get("message", "JSON-RPC error")
                data = err.get("data")
                fut.set_exception(
                    YAMSProtocolError(f"{pending.method}: {code}: {message} ({data})")
                )
            else:
                fut.set_exception(YAMSProtocolError(f"{pending.method}: {err}"))
            return

        result = msg.get("result")
        if not isinstance(result, dict):
            fut.set_exception(YAMSProtocolError(f"{pending.method}: invalid result shape"))
            return

        fut.set_result(result)

    def _fail_all_pending(self, exc: Exception) -> None:
        pending = self._pending
        self._pending = {}
        for p in pending.values():
            if not p.future.done():
                p.future.set_exception(exc)

    async def _call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_ready()
        res = await self._request(
            "tools/call",
            params={"name": name, "arguments": arguments or {}},
        )
        # res is a tool-result object, not the structured data.
        return res

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the raw tool list from MCP (best-effort)."""
        self._require_ready()
        res = await self._request("tools/list", params={})
        tools = res.get("tools")
        if isinstance(tools, list):
            return [t for t in tools if isinstance(t, dict)]
        return []

    async def refresh_tools(self) -> set[str]:
        tools = await self.list_tools()
        names: set[str] = set()
        for t in tools:
            n = t.get("name")
            if isinstance(n, str) and n:
                names.add(n)
        self._tool_names = names
        return names

    async def has_tool(self, name: str) -> bool:
        if self._tool_names is None:
            try:
                await self.refresh_tools()
            except Exception:
                return False
        return name in (self._tool_names or set())

    @staticmethod
    def _maybe_path_like(text: str) -> str | None:
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

    @staticmethod
    def _normalize_search_source(path: str, snippet: str, title: str = "") -> str:
        src = (path or "").strip()
        if src and not _is_noise_source(src):
            return src

        for candidate in (snippet, title):
            maybe = YAMSClient._maybe_path_like(candidate)
            if maybe and not _is_noise_source(maybe):
                return maybe
        return src

    async def _cli_json(self, argv: list[str], *, timeout_s: float | None = None) -> Any:
        env = os.environ.copy()
        if self._data_dir is not None:
            env["YAMS_DATA_DIR"] = self._data_dir

        proc = await asyncio.create_subprocess_exec(
            self._yams_binary,
            *argv,
            stdout=PIPE,
            stderr=PIPE,
            cwd=self._cwd or None,
            env=env,
        )
        timeout = self._request_timeout_s if timeout_s is None else float(timeout_s)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Timed out waiting for CLI response: {' '.join(argv)}")

        if proc.returncode != 0:
            detail = (
                stderr.decode("utf-8", errors="replace").strip()
                or stdout.decode("utf-8", errors="replace").strip()
            )
            raise YAMSProcessError(f"CLI command failed: {' '.join(argv)} ({detail})")

        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise YAMSProtocolError(f"CLI returned invalid JSON for {' '.join(argv)}: {e}") from e

    @staticmethod
    def _get_str(mapping: dict[str, Any], key: str) -> str:
        val = mapping.get(key)
        return val if isinstance(val, str) else ""

    @staticmethod
    def _extract_tool_data(tool_result: dict[str, Any]) -> Any:
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

    @staticmethod
    def _chunks_from_search_data(data: Any) -> list[YAMSChunk]:
        if not isinstance(data, dict):
            return []

        if isinstance(data.get("paths"), list):
            out: list[YAMSChunk] = []
            for p in data["paths"]:
                if isinstance(p, str):
                    out.append(YAMSChunk(chunk_id=p, content=p, score=0.0, source=p))
            return out

        results = data.get("results")
        if not isinstance(results, list):
            return []

        out: list[YAMSChunk] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            path = YAMSClient._get_str(r, "path")
            h = YAMSClient._get_str(r, "hash")
            rid = YAMSClient._get_str(r, "id")
            chunk_id: str = h or rid or path
            if not chunk_id:
                chunk_id = json.dumps(r, ensure_ascii=True)
            snippet = YAMSClient._get_str(r, "snippet")
            title = YAMSClient._get_str(r, "title")
            source = YAMSClient._normalize_search_source(path, snippet, title)
            meta = {k: v for k, v in r.items() if k not in {"snippet", "score"}}

            # Search anchor metadata (if MCP provides line/char spans).
            line_start = r.get("line_start")
            line_end = r.get("line_end")
            char_start = r.get("char_start")
            char_end = r.get("char_end")
            truncated = bool(r.get("snippet_truncated") or False)

            anchor_parts: list[str] = []
            if isinstance(line_start, int) and line_start > 0:
                anchor_parts.append(f"line={line_start}")
            if isinstance(line_end, int) and line_end > 0 and line_end != line_start:
                anchor_parts.append(f"line_end={line_end}")
            if isinstance(char_start, int) and char_start >= 0:
                anchor_parts.append(f"char={char_start}")
            if isinstance(char_end, int) and char_end >= 0 and char_end != char_start:
                anchor_parts.append(f"char_end={char_end}")
            if truncated:
                anchor_parts.append("truncated=true")

            anchor = ""
            if anchor_parts:
                anchor = " [" + ", ".join(anchor_parts) + "]"

            content: str = (snippet or title or source or path or chunk_id) + anchor
            score = float(r.get("score") or 0.0)
            if anchor_parts:
                score = max(score, 0.55)
            out.append(
                YAMSChunk(
                    chunk_id=chunk_id, content=content, score=score, source=source, metadata=meta
                )
            )
        return out

    @staticmethod
    def _chunks_from_list_data(data: Any) -> list[YAMSChunk]:
        if not isinstance(data, dict):
            return []
        docs = data.get("documents")
        if not isinstance(docs, list):
            return []

        out: list[YAMSChunk] = []
        for d in docs:
            if isinstance(d, str):
                out.append(YAMSChunk(chunk_id=d, content=d, score=0.0, source=d))
                continue
            if not isinstance(d, dict):
                continue
            h = YAMSClient._get_str(d, "hash")
            path = YAMSClient._get_str(d, "path")
            name = YAMSClient._get_str(d, "name")
            chunk_id: str = h or path or name
            if not chunk_id:
                chunk_id = json.dumps(d, ensure_ascii=True)
            content: str = name or path or h or chunk_id
            out.append(
                YAMSChunk(chunk_id=chunk_id, content=content, score=0.0, source=path, metadata=d)
            )
        return out

    @staticmethod
    def _parse_grep_file_paths(data: Any) -> list[tuple[str, int]]:
        """Extract unique file paths with match counts from YAMS grep output.

        Returns a list of (path, match_count) tuples sorted by match count
        descending so files with more matches (more relevant) come first.

        YAMS grep output format:
            /path/to/file.cpp (15 matches) [cpp]
              0: [R] line content here
        """
        if not isinstance(data, dict):
            return []

        # Prefer structured matches[] when available.
        structured = YAMSClient._structured_grep_matches(data)
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
            entries.sort(key=lambda e: e[1], reverse=True)
            return entries

        output = data.get("output")
        if not isinstance(output, str) or not output.strip():
            return []

        _HEADER_RE = _re.compile(r"^(/[^\s(]+)\s*\((\d+)\s+match", _re.MULTILINE)

        seen: set[str] = set()
        entries: list[tuple[str, int]] = []
        skip_exts = {".json", ".lock", ".log", ".bin", ".dat"}

        for m in _HEADER_RE.finditer(output):
            path = m.group(1)
            count = int(m.group(2))
            if path in seen:
                continue
            # Skip non-code files
            if Path(path).suffix.lower() in skip_exts:
                continue
            seen.add(path)
            entries.append((path, count))

        # If no header-style matches found, fall back to colon-delimited parsing
        if not entries:
            for ln in output.splitlines():
                ln = ln.strip()
                if not ln or ":" not in ln:
                    continue
                path = ln.split(":", 1)[0]
                if not path or path in seen:
                    continue
                if not (path.startswith("/") or path.startswith(".")):
                    continue
                if Path(path).suffix.lower() in skip_exts:
                    continue
                seen.add(path)
                entries.append((path, 1))

        # Sort by match count descending — files with more matches are more relevant
        entries.sort(key=lambda e: e[1], reverse=True)
        return entries

    @staticmethod
    def _source_matches_filters(
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

        for h in include_hints or []:
            hh = (h or "").strip()
            if not hh:
                continue
            hh_l = hh.lower()
            if any(ch in hh for ch in "*?[]"):
                if not fnmatch.fnmatch(src_l, hh_l):
                    return False
            elif hh_l not in src_l:
                return False

        for h in exclude_hints or []:
            hh = (h or "").strip()
            if not hh:
                continue
            hh_l = hh.lower()
            if any(ch in hh for ch in "*?[]"):
                if fnmatch.fnmatch(src_l, hh_l):
                    return False
            elif hh_l in src_l:
                return False

        return True

    @staticmethod
    def _structured_grep_matches(data: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(data, dict):
            return {}
        raw = data.get("matches")
        if not isinstance(raw, list):
            return {}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = YAMSClient._get_str(item, "file")
            if not path:
                continue
            grouped.setdefault(path, []).append(item)
        return grouped

    @staticmethod
    def _read_file_context(
        filepath: str,
        pattern: str,
        *,
        context_lines: int = 10,
        max_matches: int = 10,
        max_chars: int = 4000,
    ) -> str | None:
        """Read a file from disk and extract context windows around pattern matches.

        Returns a formatted string with match regions, or None if the file
        can't be read or the pattern doesn't match.
        """
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

        # Find matching line numbers
        try:
            pat = _re.compile(pattern, _re.IGNORECASE)
        except _re.error:
            # Fall back to literal substring search
            pat = None

        match_line_nums: list[int] = []
        for i, line in enumerate(lines):
            if pat is not None:
                if pat.search(line):
                    match_line_nums.append(i)
            elif pattern.lower() in line.lower():
                match_line_nums.append(i)
            if len(match_line_nums) >= max_matches:
                break

        if not match_line_nums:
            return None

        # Build context windows, merging overlapping regions
        regions: list[tuple[int, int]] = []
        for lno in match_line_nums:
            start = max(0, lno - context_lines)
            end = min(len(lines), lno + context_lines + 1)
            if regions and start <= regions[-1][1]:
                # Merge with previous region
                regions[-1] = (regions[-1][0], end)
            else:
                regions.append((start, end))

        # Format output with line numbers
        parts: list[str] = []
        total_chars = 0
        for start, end in regions:
            region_lines: list[str] = []
            for i in range(start, end):
                numbered = f"{i + 1:>5}: {lines[i]}"
                region_lines.append(numbered)
            region_text = "\n".join(region_lines)
            total_chars += len(region_text)
            if total_chars > max_chars:
                # Truncate this region to fit budget
                remaining = max_chars - (total_chars - len(region_text))
                if remaining > 100:
                    parts.append(region_text[:remaining] + "\n  ... [truncated]")
                break
            parts.append(region_text)

        if not parts:
            return None

        return "\n  ---\n".join(parts)

    @staticmethod
    def _enrich_grep_results(
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
        """Parse grep output, read matched files from disk, return enriched chunks.

        Each chunk contains actual code context around match regions, not just
        file paths.  Falls back to path-only chunks for files that can't be read.
        Files are sorted by match count descending so the most relevant files
        (with the most matches) are processed first.
        """
        structured = YAMSClient._structured_grep_matches(data)
        if structured:
            return YAMSClient._chunks_from_structured_grep_matches(
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

        file_entries = YAMSClient._parse_grep_file_paths(data)
        if not file_entries:
            return []

        if path_hint:
            hint = path_hint.strip()
            if hint:
                if "/" in hint:
                    filtered = [e for e in file_entries if hint in e[0]]
                else:
                    filtered = [e for e in file_entries if Path(e[0]).name == hint]
                if filtered:
                    file_entries = filtered

        file_entries = [
            e
            for e in file_entries
            if YAMSClient._source_matches_filters(
                e[0],
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

        for i, (fpath, fmatches) in enumerate(file_entries[:max_files]):
            context = YAMSClient._read_file_context(
                fpath,
                pattern,
                context_lines=tuned_context_lines,
                max_matches=tuned_max_matches,
                max_chars=tuned_max_chars,
            )
            # Score higher for files with more matches
            base_score = (
                min(1.0, 0.3 + (fmatches / max(1, match_count)) * 0.7) if match_count else 1.0
            )
            if context:
                # Rich chunk with actual code context
                header = f"# {fpath} ({fmatches} matches)\n"
                content = header + context
                chunks.append(
                    YAMSChunk(
                        chunk_id=f"grep:{i}:{Path(fpath).name}:{pattern_tag}",
                        content=content,
                        score=base_score,
                        source=fpath,
                        metadata={
                            "match_count": match_count,
                            "file_count": file_count,
                            "file_matches": fmatches,
                            "enriched": True,
                        },
                    )
                )
                logger.debug(
                    "grep enriched %s: %d matches, %d chars of context",
                    fpath,
                    fmatches,
                    len(content),
                )
            else:
                # Fallback: path-only chunk (file not on disk or pattern mismatch)
                chunks.append(
                    YAMSChunk(
                        chunk_id=f"grep:{i}:{Path(fpath).name}:{pattern_tag}",
                        content=f"[file: {fpath}]",
                        score=0.3,  # lower score for path-only
                        source=fpath,
                        metadata={
                            "match_count": match_count,
                            "file_count": file_count,
                            "enriched": False,
                        },
                    )
                )
                logger.debug("grep fallback (path only) %s", fpath)

        return chunks

    @staticmethod
    def _chunks_from_structured_grep_matches(
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
                    entries = [e for e in entries if hint in e[0]]
                else:
                    entries = [e for e in entries if Path(e[0]).name == hint]

        entries = [
            e
            for e in entries
            if YAMSClient._source_matches_filters(
                e[0],
                cwd=cwd,
                path_hint=path_hint,
                include_hints=include_hints,
                exclude_hints=exclude_hints,
            )
        ]

        entries.sort(key=lambda e: e[1], reverse=True)
        if not entries:
            return []

        pattern_tag = hashlib.md5((pattern or "").encode("utf-8")).hexdigest()[:8]
        chunks: list[YAMSChunk] = []
        pat_terms = [t.lower() for t in _re.findall(r"[A-Za-z0-9_]+", pattern or "") if len(t) >= 3]

        for i, (fpath, fmatches) in enumerate(entries[:max_files]):
            matches = grouped_matches.get(fpath, [])
            rendered: list[str] = []
            total_chars = 0
            has_structured_context = False

            for m in matches:
                line_text = YAMSClient._get_str(m, "line_text")
                before = (
                    m.get("context_before") if isinstance(m.get("context_before"), list) else []
                )
                after = m.get("context_after") if isinstance(m.get("context_after"), list) else []
                try:
                    lno = int(m.get("line_number") or 0)
                except Exception:
                    lno = 0
                if line_text.strip() or before or after or lno > 0:
                    has_structured_context = True
                    break

            if has_structured_context:
                for m in matches:
                    try:
                        lno = int(m.get("line_number") or 0)
                    except Exception:
                        lno = 0
                    line_text = YAMSClient._get_str(m, "line_text")
                    before = (
                        m.get("context_before") if isinstance(m.get("context_before"), list) else []
                    )
                    after = (
                        m.get("context_after") if isinstance(m.get("context_after"), list) else []
                    )

                    if not (line_text.strip() or before or after or lno > 0):
                        continue

                    if before:
                        for b in before[-context_lines:]:
                            rendered.append(f"      {str(b)}")

                    if lno > 0:
                        rendered.append(f"{lno:>5}: {line_text}")
                    else:
                        rendered.append(f"      {line_text}")

                    if after:
                        for a in after[:context_lines]:
                            rendered.append(f"      {str(a)}")

                    rendered.append("  ---")
                    total_chars = sum(len(x) + 1 for x in rendered)
                    if total_chars >= max_chars_per_file:
                        break
            else:
                fallback_ctx = YAMSClient._read_file_context(
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

            # Signal quality: prefer files where matched lines include query terms.
            term_hits = 0
            term_total = max(1, len(pat_terms))
            if pat_terms:
                lower_body = body.lower()
                term_hits = sum(1 for t in pat_terms if t in lower_body)
            completeness = min(1.0, (term_hits / term_total) if pat_terms else 0.6)
            score = min(1.0, 0.25 + min(0.55, fmatches / 20.0) + (0.20 * completeness))

            header = f"# {fpath} ({fmatches} matches)\n"
            content = header + body if body else header
            chunks.append(
                YAMSChunk(
                    chunk_id=f"grep:{i}:{Path(fpath).name}:{pattern_tag}",
                    content=content,
                    score=score,
                    source=fpath,
                    metadata={
                        "file_matches": fmatches,
                        "term_hits": term_hits,
                        "term_total": term_total,
                        "enriched": True,
                        "structured": True,
                    },
                )
            )

        return chunks

    @staticmethod
    def _single_chunk_json(kind: str, data: Any) -> list[YAMSChunk]:
        try:
            text = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)
        except Exception:
            text = str(data)
        return [YAMSChunk(chunk_id=kind, content=text, score=0.0, source=kind, metadata={})]

    @staticmethod
    def _search_result_quality(chunks: list[YAMSChunk], query: str) -> float:
        if not chunks:
            return 0.0

        code_exts = {
            ".cpp",
            ".c",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".py",
            ".rs",
            ".ts",
            ".js",
            ".go",
            ".java",
        }
        q_l = (query or "").lower()
        q_file = None
        m = _re.search(r"([A-Za-z_][\w\-]*\.[A-Za-z0-9]{1,8})", q_l)
        if m:
            q_file = m.group(1)

        total = float(len(chunks))
        mean_score = sum(float(c.score or 0.0) for c in chunks) / total
        code_ratio = 0.0
        anchor_ratio = 0.0
        noise_ratio = 0.0
        file_match_bonus = 0.0

        for c in chunks:
            src = (c.source or "").strip()
            src_l = src.lower()
            if src:
                if Path(src).suffix.lower() in code_exts:
                    code_ratio += 1.0
                if _is_noise_source(src):
                    noise_ratio += 1.0
                if q_file and q_file in src_l:
                    file_match_bonus = 0.15
            txt = (c.content or "").lower()
            if "line=" in txt or "char=" in txt or "truncated=true" in txt:
                anchor_ratio += 1.0

        code_ratio /= total
        anchor_ratio /= total
        noise_ratio /= total

        quality = (
            (0.35 * mean_score)
            + (0.35 * code_ratio)
            + (0.20 * anchor_ratio)
            + (0.10 * (1.0 - noise_ratio))
            + file_match_bonus
        )
        return max(0.0, min(1.0, quality))

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[YAMSChunk]:
        args = {"query": query, "limit": limit}

        # Scope to cwd if configured and not overridden by caller.
        if self._cwd and "cwd" not in kwargs:
            args["cwd"] = self._cwd

        args.update(kwargs)

        # Prefer hybrid by default; only fallback to keyword when hybrid has
        # no useful coverage. This avoids doubling MCP traffic on every search.
        requested_type = str(args.get("type") or "").strip().lower()
        tried_types: list[str] = [requested_type] if requested_type else ["hybrid", "keyword"]
        last_err: Exception | None = None
        fallback_quality_threshold = 0.30

        candidates: list[tuple[float, str, list[YAMSChunk]]] = []
        for t in tried_types:
            run_args = dict(args)
            run_args["type"] = t
            try:
                tool_result = await self._call_tool("search", run_args)
                data = self._extract_tool_data(tool_result)
                chunks = self._chunks_from_search_data(data)
                # Filter out obvious out-of-scope files early.
                chunks = [
                    c
                    for c in chunks
                    if not c.source or self._source_matches_filters(c.source, cwd=self._cwd)
                ]
                self._backfill_positional_scores(chunks)
                q = self._search_result_quality(chunks, query)
                candidates.append((q, t, chunks))

                if not requested_type and t == "hybrid":
                    if chunks and q >= fallback_quality_threshold:
                        return chunks
            except Exception as e:
                last_err = e
                try:
                    cli_args = ["search", query, "--limit", str(limit), "--json"]
                    if t:
                        cli_args.extend(["--type", t])
                    data = await self._cli_json(
                        cli_args, timeout_s=max(30.0, self._request_timeout_s)
                    )
                    chunks = self._chunks_from_search_data(data)
                    chunks = [
                        c
                        for c in chunks
                        if not c.source or self._source_matches_filters(c.source, cwd=self._cwd)
                    ]
                    self._backfill_positional_scores(chunks)
                    q = self._search_result_quality(chunks, query)
                    candidates.append((q, f"cli:{t}", chunks))
                    if not requested_type and t == "hybrid":
                        if chunks and q >= fallback_quality_threshold:
                            return chunks
                except Exception:
                    pass
                continue

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][2]

        if last_err is not None:
            raise last_err
        return []

    @staticmethod
    def _backfill_positional_scores(chunks: list[YAMSChunk]) -> None:
        """Assign descending positional scores when YAMS returns all-zero scores."""
        if not chunks:
            return
        if any(c.score > 0.0 for c in chunks):
            return  # real scores present; leave as-is
        n = len(chunks)
        for i, c in enumerate(chunks):
            # Linear decay: first result = 1.0, last = max(0.1, 1/n)
            c.score = max(0.1, 1.0 - (i / max(1, n)))

    async def grep(self, pattern: str, **kwargs: Any) -> list[YAMSChunk]:
        pat, path_hint, include_hints, exclude_hints = self._split_grep_pattern(pattern)
        pat_l = (pat or "").lower()
        if not exclude_hints and not any(k in pat_l for k in ("test", "bench", "doc")):
            exclude_hints = ["*/tests/*", "*/docs/*", "*/benchmarks/*"]
        args = {"pattern": pat}
        # Scope to cwd if configured and not overridden by caller.
        if self._cwd and "cwd" not in kwargs:
            args["cwd"] = self._cwd
        args.update(kwargs)
        try:
            tool_result = await self._call_tool("grep", args)
            data = self._extract_tool_data(tool_result)
        except Exception:
            data = await self._cli_json(
                ["grep", pat, "--json"], timeout_s=max(30.0, self._request_timeout_s)
            )
        chunks = self._enrich_grep_results(
            data,
            pat,
            cwd=self._cwd,
            path_hint=path_hint,
            include_hints=include_hints,
            exclude_hints=exclude_hints,
        )
        # Assign positional scores so the assembler can rank earlier
        # (more relevant) matches higher.
        self._backfill_positional_scores(chunks)
        return chunks

    @staticmethod
    def _split_grep_pattern(pattern: str) -> tuple[str, str | None, list[str], list[str]]:
        """Split a grep pattern from an optional path hint.

        Accepts formats like:
          "registerTool path:mcp_server.cpp"
          "registerTool path:src/mcp/mcp_server.cpp"
          "registerTool include:src/mcp/* exclude:tests/*"
        """
        raw = (pattern or "").strip()
        if not raw:
            return "", None, [], []

        def _extract(token_pattern: str) -> list[str]:
            vals = [m.group(1).strip().strip("'\"") for m in _re.finditer(token_pattern, raw)]
            out: list[str] = []
            for v in vals:
                for p in v.split(","):
                    q = p.strip()
                    if q and q not in out:
                        out.append(q)
            return out

        path_vals = _extract(r"\bpath:([^\s]+)")
        include_vals = _extract(r"\binclude:([^\s]+)")
        exclude_vals = _extract(r"\bexclude:([^\s]+)")

        cleaned = raw
        cleaned = _re.sub(r"\s*\bpath:[^\s]+", "", cleaned)
        cleaned = _re.sub(r"\s*\binclude:[^\s]+", "", cleaned)
        cleaned = _re.sub(r"\s*\bexclude:[^\s]+", "", cleaned)
        cleaned = cleaned.strip()

        return cleaned or raw, (path_vals[0] if path_vals else None), include_vals, exclude_vals

    async def graph(self, query: str) -> list[YAMSChunk]:
        q = (query or "").strip()
        args = self._parse_graph_query(q)
        data = await self.graph_query(**args)
        chunks = self._chunks_from_graph_data(data, query=q)
        self._backfill_positional_scores(chunks)
        return chunks

    def _parse_graph_query(self, raw_query: str) -> dict[str, Any]:
        q = (raw_query or "").strip()
        opts: dict[str, Any] = {"depth": 2, "limit": 40}
        if not q:
            return opts

        def _extract_int(key: str) -> None:
            m = _re.search(rf"\b{key}:(\d+)", q, flags=_re.IGNORECASE)
            if not m:
                return
            try:
                opts[key] = max(1, int(m.group(1)))
            except Exception:
                return

        _extract_int("depth")
        _extract_int("limit")

        mrel = _re.search(r"\brelation:([^\s]+)", q, flags=_re.IGNORECASE)
        if mrel:
            opts["relation"] = mrel.group(1).strip().strip("'\"")

        mrev = _re.search(r"\breverse:(true|false|1|0)", q, flags=_re.IGNORECASE)
        if mrev:
            opts["reverse"] = mrev.group(1).lower() in {"true", "1"}

        cleaned = q
        cleaned = _re.sub(r"\s*\bdepth:\d+", "", cleaned, flags=_re.IGNORECASE)
        cleaned = _re.sub(r"\s*\blimit:\d+", "", cleaned, flags=_re.IGNORECASE)
        cleaned = _re.sub(r"\s*\brelation:[^\s]+", "", cleaned, flags=_re.IGNORECASE)
        cleaned = _re.sub(
            r"\s*\breverse:(?:true|false|1|0)", "", cleaned, flags=_re.IGNORECASE
        ).strip()

        if cleaned.startswith("node_key:"):
            opts["node_key"] = cleaned.split(":", 1)[1].strip().strip("'\"")
            return opts

        if self._looks_like_hash(cleaned):
            opts["hash"] = cleaned
            return opts

        # Path-like query: resolve to graph node key for file nodes.
        if "/" in cleaned or cleaned.endswith((".cpp", ".h", ".hpp", ".py", ".rs", ".ts", ".js")):
            try:
                p = Path(cleaned)
                if not p.is_absolute() and self._cwd:
                    p = Path(self._cwd) / p
                abs_path = str(p.resolve())
                opts["node_key"] = f"path:file:{abs_path}"
                return opts
            except Exception:
                pass

        opts["name"] = cleaned
        return opts

    @staticmethod
    def _chunks_from_graph_data(data: Any, *, query: str = "") -> list[YAMSChunk]:
        if not isinstance(data, dict):
            return YAMSClient._single_chunk_json("graph", data)

        nodes = data.get("connected_nodes")
        if not isinstance(nodes, list):
            return YAMSClient._single_chunk_json("graph", data)

        out: list[YAMSChunk] = []
        type_counts = (
            data.get("node_type_counts") if isinstance(data.get("node_type_counts"), dict) else {}
        )
        summary = (
            f"graph total_nodes={data.get('total_nodes_found', 0)} "
            f"edges={data.get('total_edges_traversed', 0)} depth={data.get('max_depth_reached', 0)}"
        )
        out.append(
            YAMSChunk(
                chunk_id="graph:summary",
                content=summary,
                score=0.20,
                source="graph",
                metadata={"type_counts": type_counts, "query": query},
            )
        )

        for i, n in enumerate(nodes):
            if not isinstance(n, dict):
                continue
            label = YAMSClient._get_str(n, "label")
            ntype = YAMSClient._get_str(n, "type")
            nkey = YAMSClient._get_str(n, "nodeKey")
            dist = n.get("distance")
            if isinstance(dist, int):
                d = max(0, dist)
            elif isinstance(dist, float):
                d = max(0, int(dist))
            elif isinstance(dist, str):
                try:
                    d = max(0, int(dist.strip() or "0"))
                except Exception:
                    d = 0
            else:
                d = 0
            score = max(0.10, 0.36 - (0.07 * d))
            source = label if "/" in label else (nkey or "graph")
            text = f"[{ntype}] {label} (distance={d}) key={nkey}"
            out.append(
                YAMSChunk(
                    chunk_id=f"graph:{i}:{nkey or label or i}",
                    content=text,
                    score=score,
                    source=source,
                    metadata=n,
                )
            )

        return out

    async def graph_query(self, **kwargs: Any) -> dict[str, Any]:
        """Low-level graph tool call with full parameter control.

        Accepts any parameters the YAMS graph tool supports:
          - list_types: bool (return node type counts)
          - list_type: str (list nodes of a specific type)
          - node_key: str (e.g. "file:/path/to/file.cpp")
          - name: str, hash: str, node_id: int
          - relation: str (filter to specific edge type)
          - depth: int (BFS depth, 1-5)
          - reverse: bool (walk incoming edges)
          - limit: int, offset: int
          - include_properties: bool
          - isolated: bool (find disconnected nodes)
          - action: str ("query" or "ingest")

        Returns the raw structured data dict from the tool response.
        """
        tool_result = await self._call_tool("graph", kwargs)
        data = self._extract_tool_data(tool_result)
        if isinstance(data, dict):
            return data
        return {"raw": data}

    async def get(self, name_or_hash: str) -> YAMSChunk | None:
        q = (name_or_hash or "").strip()
        try_hash_first = self._looks_like_hash(q)

        def hash_args() -> dict[str, Any]:
            return {"hash": q, "include_content": True}

        def name_args() -> dict[str, Any]:
            return {"name": q, "include_content": True}

        try:
            tool_result = await self._call_tool(
                "get", hash_args() if try_hash_first else name_args()
            )
        except YAMSProtocolError as e:
            msg = str(e).lower()
            not_found = "not found" in msg or "missing" in msg
            if not not_found:
                raise
            # Fallback: try the other selector.
            tool_result = await self._call_tool(
                "get", name_args() if try_hash_first else hash_args()
            )

        data = self._extract_tool_data(tool_result)
        if not isinstance(data, dict):
            return None
        content: str = self._get_str(data, "content")
        h = self._get_str(data, "hash")
        path = self._get_str(data, "path")
        name = self._get_str(data, "name")
        chunk_id: str = h or path or name or q
        source: str = path
        return YAMSChunk(
            chunk_id=chunk_id, content=content, score=0.0, source=source, metadata=data
        )

    async def list_docs(self, **kwargs: Any) -> list[YAMSChunk]:
        tool_result = await self._call_tool("list", kwargs)
        data = self._extract_tool_data(tool_result)
        return self._chunks_from_list_data(data)

    async def add(
        self,
        content: str,
        name: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        args: dict[str, Any] = {"content": content, "name": name}
        if tags is not None:
            args["tags"] = tags
        if metadata is not None:
            args["metadata"] = metadata

        tool_result = await self._call_tool("add", args)
        data = self._extract_tool_data(tool_result)
        if isinstance(data, dict) and isinstance(data.get("hash"), str) and data["hash"]:
            return data["hash"]
        raise YAMSProtocolError(f"add: unexpected response: {data}")

    async def status(self) -> dict[str, Any]:
        try:
            tool_result = await self._call_tool("status", {})
            data = self._extract_tool_data(tool_result)
        except Exception:
            data = await self._cli_json(
                ["status", "--json"], timeout_s=max(30.0, self._request_timeout_s)
            )
        if isinstance(data, dict):
            return data
        return {"data": data}

    async def pipeline(self, steps: list[dict[str, Any]]) -> list[Any]:
        """Execute a multi-step read-only pipeline.

        Steps format: [{"op": "search", "params": {...}}, ...]
        Supports "$prev" expansion (server-side when the composite "query" tool exists;
        client-side fallback otherwise).
        """

        # Prefer server-side pipeline if available.
        if await self.has_tool("query"):
            try:
                tool_result = await self._call_tool("query", {"steps": steps})
                data = self._extract_tool_data(tool_result)
                return self._pipeline_results_from_query_data(data)
            except YAMSProtocolError as e:
                msg = str(e)
                if (
                    "Unknown tool" not in msg
                    and "Method not found" not in msg
                    and "not found" not in msg
                ):
                    raise
                logger.debug("Falling back to client-side pipeline: %s", e)

        return await self._pipeline_client_side(steps)

    @staticmethod
    def _pipeline_results_from_query_data(data: Any) -> list[Any]:
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            out: list[Any] = []
            for step in data["steps"]:
                if isinstance(step, dict) and "result" in step:
                    out.append(step["result"])
            return out
        return [data]

    async def _pipeline_client_side(self, steps: list[dict[str, Any]]) -> list[Any]:
        prev: Any = {}
        results: list[Any] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "op" not in step:
                raise YAMSProtocolError(f"pipeline: step {i} missing 'op'")
            op = step["op"]
            params = step.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params = self._resolve_prev_refs(params, prev)
            tool_result = await self._call_tool(str(op), params)
            prev = self._extract_tool_data(tool_result)
            results.append(prev)
        return results

    @staticmethod
    def _resolve_prev_refs(params: dict[str, Any], prev: Any) -> dict[str, Any]:
        # Mirror the server-side rules: replace string values starting with "$prev".
        def resolve_one(s: str) -> Any:
            if s == "$prev":
                return prev
            if not s.startswith("$prev"):
                return s
            cur: Any = prev
            path = s[5:]
            while path and cur is not None:
                if path.startswith("."):
                    path = path[1:]
                    end = len(path)
                    for j, ch in enumerate(path):
                        if ch in ".[":
                            end = j
                            break
                    key = path[:end]
                    path = path[end:]
                    if isinstance(cur, dict):
                        cur = cur.get(key)
                    else:
                        return None
                elif path.startswith("["):
                    close = path.find("]")
                    if close == -1:
                        return None
                    idx_str = path[1:close]
                    path = path[close + 1 :]
                    try:
                        idx = int(idx_str)
                    except Exception:
                        return None
                    if isinstance(cur, list) and 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return None
                else:
                    return None
            return cur

        resolved: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$prev"):
                resolved[k] = resolve_one(v)
            else:
                resolved[k] = v
        return resolved

    async def execute_spec(self, spec: QuerySpec) -> YAMSQueryResult:
        t0 = time.perf_counter()
        try:
            if spec.query_type == QueryType.SEMANTIC:
                chunks = await self.search(spec.query)
            elif spec.query_type == QueryType.GREP:
                chunks = await self.grep(spec.query)
            elif spec.query_type == QueryType.GRAPH:
                chunks = await self.graph(spec.query)
            elif spec.query_type == QueryType.GET:
                ch = await self.get(spec.query)
                chunks = [ch] if ch is not None else []
            elif spec.query_type == QueryType.LIST:
                chunks = await self.list_docs(pattern=spec.query, limit=50)
            else:
                raise ValueError(f"Unknown QueryType: {spec.query_type}")

            return YAMSQueryResult(
                spec=spec,
                chunks=chunks,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=None,
            )
        except Exception as e:
            return YAMSQueryResult(
                spec=spec,
                chunks=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(e),
            )
