from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from dcs.lmstudio_context import preload_model
from dcs.pipeline import DCSPipeline
from dcs.types import PipelineConfig, PipelineResult


@dataclass
class RoutingPolicy:
    quality_threshold: float = 0.7
    relaxed_quality_floor: float = 0.55
    strict_quality_margin: float = 0.1
    score_threshold: float = 0.68
    min_sources: int = 1
    min_output_chars: int = 40
    min_task_term_coverage: float = 0.3
    require_non_error_output: bool = True
    preload_tier_models: bool = False
    preload_retries: int = 2
    preload_retry_backoff_s: float = 2.0


@dataclass
class RoutingOutcome:
    selected_tier: int
    selected_result: PipelineResult
    tier_results: list[PipelineResult] = field(default_factory=list)

    @property
    def escalated(self) -> bool:
        return self.selected_tier > 0


class TieredRouter:
    """Run DCS with escalation tiers and select the best acceptable result."""

    def __init__(
        self,
        base_config: PipelineConfig,
        fallback_configs: list[PipelineConfig] | None = None,
        policy: RoutingPolicy | None = None,
        pipeline_factory: Callable[[PipelineConfig], DCSPipeline] = DCSPipeline,
    ):
        self.base_config = base_config
        self.fallback_configs = list(fallback_configs or [])
        self.policy = policy or RoutingPolicy(
            quality_threshold=float(base_config.quality_threshold or 0.7),
            min_sources=1,
            require_non_error_output=True,
        )
        self.pipeline_factory = pipeline_factory

    async def run(self, task: str) -> RoutingOutcome:
        configs = [self.base_config, *self.fallback_configs]
        results: list[PipelineResult] = []

        best_idx = 0
        best_score = -1.0

        for idx, cfg in enumerate(configs):
            if idx > 0 and self.policy.preload_tier_models:
                self._preload_tier(cfg)
            pipe = self.pipeline_factory(cfg)
            res = await pipe.run(task)
            results.append(res)

            score = self._score(res)
            if score > best_score:
                best_idx = idx
                best_score = score

            if self._accept(res):
                return RoutingOutcome(selected_tier=idx, selected_result=res, tier_results=results)

        return RoutingOutcome(
            selected_tier=best_idx, selected_result=results[best_idx], tier_results=results
        )

    def _accept(self, res: PipelineResult) -> bool:
        crit = res.final_critique
        quality = float(crit.quality_score if crit else 0.0)
        out = (res.final_output or "").strip()
        if self.policy.require_non_error_output and (not out or out.startswith("Error")):
            return False

        if len(out) < int(self.policy.min_output_chars):
            return False

        sources = self._source_count(res)
        if sources < int(self.policy.min_sources):
            return False

        hard_accept = float(self.policy.quality_threshold) + float(
            self.policy.strict_quality_margin
        )
        if quality >= hard_accept:
            return True

        # For medium-confidence outputs (including exact-threshold cases),
        # require grounding checks before accepting without escalation.
        if quality < float(self.policy.relaxed_quality_floor):
            return False

        coverage = self._task_term_coverage(res.task, out)
        if coverage < float(self.policy.min_task_term_coverage):
            return False

        if self._score(res) < float(self.policy.score_threshold):
            return False

        return True

    @staticmethod
    def _task_term_coverage(task: str, output: str) -> float:
        terms = TieredRouter._extract_task_terms(task)
        if not terms:
            return 0.0
        out_l = (output or "").lower()
        hit = sum(1 for t in terms if t in out_l)
        return hit / max(1, len(terms))

    @staticmethod
    def _extract_task_terms(task: str) -> list[str]:
        stop = {
            "what",
            "how",
            "does",
            "with",
            "from",
            "that",
            "this",
            "into",
            "over",
            "when",
            "where",
            "which",
            "using",
            "used",
            "list",
            "focus",
            "server",
            "client",
        }
        raw = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", task or "")]
        terms: list[str] = []
        for w in raw:
            if len(w) < 4 or w in stop:
                continue
            if w not in terms:
                terms.append(w)
            if len(terms) >= 10:
                break
        return terms

    def _source_count(self, res: PipelineResult) -> int:
        seen: set[str] = set()
        for it in res.iterations:
            if it.context:
                for s in it.context.sources:
                    if s:
                        seen.add(s)
            for qr in it.query_results:
                for ch in qr.chunks:
                    if ch.source:
                        seen.add(ch.source)
        return len(seen)

    def _score(self, res: PipelineResult) -> float:
        crit = res.final_critique
        quality = float(crit.quality_score if crit else 0.0)
        cov = min(1.0, self._source_count(res) / 10.0)
        out = (res.final_output or "").strip()
        non_error = 0.0 if (not out or out.startswith("Error")) else 1.0
        return 0.75 * quality + 0.2 * cov + 0.05 * non_error

    def _preload_tier(self, cfg: PipelineConfig) -> None:
        try:
            preload_model(
                cfg.executor_model.name,
                base_url=cfg.executor_model.base_url,
                api_key=cfg.executor_model.api_key,
                context_length=int(cfg.executor_model.context_window),
                min_ready_context_length=65535,
                keep_model_in_memory=True,
                retries=int(self.policy.preload_retries),
                retry_backoff_s=float(self.policy.preload_retry_backoff_s),
                ready_timeout_s=max(300.0, float(cfg.executor_model.request_timeout_s or 600.0)),
                ready_poll_s=max(1.0, float(self.policy.preload_retry_backoff_s)),
                required_successes=2,
            )
            crit = cfg.critic_model
            if crit and crit.name != cfg.executor_model.name:
                preload_model(
                    crit.name,
                    base_url=crit.base_url,
                    api_key=crit.api_key,
                    context_length=int(crit.context_window),
                    min_ready_context_length=65535,
                    keep_model_in_memory=True,
                    retries=int(self.policy.preload_retries),
                    retry_backoff_s=float(self.policy.preload_retry_backoff_s),
                    ready_timeout_s=max(300.0, float(crit.request_timeout_s or 600.0)),
                    ready_poll_s=max(1.0, float(self.policy.preload_retry_backoff_s)),
                    required_successes=2,
                )
        except Exception:
            pass
