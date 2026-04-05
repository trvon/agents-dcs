from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeSettings:
    config_path: Path | None = None
    config_dir: Path | None = None
    task_dir: Path | None = None
    yams_cwd: Path | None = None
    critic_debug_dir: Path | None = None


def _resolve_relative(config_path: Path, raw: str | None) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def find_runtime_config(base_dir: Path) -> Path | None:
    env_path = os.environ.get("DCS_CONFIG_TOML", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(base_dir / "config.toml")
    candidates.append(Path.cwd() / "config.toml")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_runtime_settings(base_dir: Path) -> RuntimeSettings:
    cfg_path = find_runtime_config(base_dir)
    if cfg_path is None:
        return RuntimeSettings()

    raw = _load_toml(cfg_path)
    paths = raw.get("paths") or {}
    debug = raw.get("debug") or {}
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(debug, dict):
        debug = {}

    return RuntimeSettings(
        config_path=cfg_path,
        config_dir=_resolve_relative(cfg_path, paths.get("config_dir")),
        task_dir=_resolve_relative(cfg_path, paths.get("task_dir")),
        yams_cwd=_resolve_relative(cfg_path, paths.get("yams_cwd")),
        critic_debug_dir=_resolve_relative(cfg_path, debug.get("critic_debug_dir")),
    )
