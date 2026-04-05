#!/usr/bin/env python3
"""End-to-end smoke test for the DCS pipeline.

Runs a single task through the full pipeline:
  decompose → retrieve → assemble → execute → critique

Usage:
    uv run python smoke_test.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dcs.pipeline import DCSPipeline
from dcs.runtime_config import load_runtime_settings
from dcs.types import ModelConfig, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


TASK = (
    "What MCP tools does the YAMS server expose? "
    "List each tool name and a one-line description of what it does. "
    "Focus on the tool names registered in mcp_server.cpp."
)


def _default_yams_cwd() -> str:
    base_dir = Path(__file__).resolve().parent
    runtime = load_runtime_settings(base_dir)
    env_cwd = os.environ.get("YAMS_CWD", "").strip()
    if env_cwd:
        return env_cwd
    if runtime.yams_cwd is not None:
        return str(runtime.yams_cwd)
    # external/agent/smoke_test.py -> external -> yams
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "src").exists():
        return str(repo_root)
    return str(Path.cwd())


async def main() -> int:
    model = ModelConfig(
        name="google/gemma-3-27b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        context_window=16384,
        max_output_tokens=2048,
        temperature=0.7,
    )

    # Use a non-thinking model for critique to avoid wasting output tokens on
    # reasoning and to get reliable JSON output.
    critic_model = ModelConfig(
        name="openai/gpt-oss-20b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        context_window=16384,
        max_output_tokens=512,
        temperature=0.3,
    )

    config = PipelineConfig(
        executor_model=model,
        critic_model=critic_model,
        context_budget=10000,
        max_queries_per_iteration=4,
        max_chunks_per_query=8,
        min_chunk_score=0.0,  # keyword search scores are 0-based
        max_iterations=10,
        quality_threshold=0.7,
        convergence_delta=0.01,
        yams_cwd=_default_yams_cwd(),
    )

    pipe = DCSPipeline(config)

    print("\n=== DCS Pipeline Smoke Test ===\n", file=sys.stderr)
    result = await pipe.run(TASK)

    print("\n=== Final Output ===\n", file=sys.stderr)
    output = result.final_output or "(no output)"
    print(f"[len={len(output)}, repr={repr(output[:200])}]", file=sys.stderr)
    print(output)

    print("\n=== Summary ===", file=sys.stderr)
    print(f"Iterations: {result.num_iterations}", file=sys.stderr)
    print(f"Converged:  {result.converged}", file=sys.stderr)
    print(f"Latency:    {result.total_latency_ms:.0f}ms", file=sys.stderr)

    crit = result.final_critique
    if crit:
        print(f"Quality:    {crit.quality_score:.2f}", file=sys.stderr)
        print(f"Missing:    {crit.missing_info}", file=sys.stderr)
        print(f"Irrelevant: {len(crit.irrelevant_chunks)} chunks", file=sys.stderr)

    # Non-zero exit if no output produced
    if not result.final_output or result.final_output.startswith("Error"):
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        return 1

    print("\nSMOKE TEST PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
