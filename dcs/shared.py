from __future__ import annotations

from dcs.types import QuerySpec


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def is_noise_source(path: str) -> bool:
    p = (path or "").lower()
    if not p:
        return False
    if "/tests/" in p or "/docs/" in p or "/benchmarks/" in p:
        return True
    return p.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".lock"))


def spec_key(spec: QuerySpec) -> tuple[str, str]:
    return (spec.query_type.value, spec.query.strip())
