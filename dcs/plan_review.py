from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dcs.assembler import ContextAssembler
from dcs.client import YAMSClient
from dcs.critic import _extract_first_json_object, _try_parse_json
from dcs.executor import ModelExecutor
from dcs.types import (
    ContextBlock,
    ModelConfig,
    PipelineConfig,
    PlanReviewInput,
    PlanReviewResult,
    PlanStep,
    PlanStepReview,
    PlanStepStatus,
    QuerySpec,
    QueryType,
    YAMSChunk,
    YAMSQueryResult,
)

logger = logging.getLogger(__name__)


_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>", re.IGNORECASE)
_ORPHAN_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>[\s\S]*$", re.IGNORECASE)
_PLAN_SECTION_TYPES = {
    "assumption": "assumption",
    "assumptions": "assumption",
    "tests first": "required_step",
    "minimum safe change set": "required_step",
    "benchmark cases to rerun": "benchmark",
    "benchmark cases": "benchmark",
    "acceptance gates": "acceptance_gate",
    "what's covered now": "evidence",
    "what is covered now": "evidence",
    "what still blocks": "gap",
    "what still blocks end-to-end confidence": "gap",
    "what's still blocked": "gap",
    "what is still blocked": "gap",
    "what we decide after that": "decision_branch",
}

_NON_EXECUTABLE_STEP_TYPES = {"evidence", "gap", "decision_branch"}


def _clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def _strip_system_reminders(text: str) -> str:
    cleaned = _SYSTEM_REMINDER_RE.sub("\n", text or "")
    cleaned = _ORPHAN_SYSTEM_REMINDER_RE.sub("\n", cleaned)
    return cleaned.strip()


def _canonical_section_name(line: str) -> str:
    raw = re.sub(r":+$", "", (line or "").strip())
    key = raw.replace("’", "'").replace("`", "'")
    key = re.sub(r"\s+", " ", key).strip().lower()
    if key in _PLAN_SECTION_TYPES:
        return raw.strip()
    return ""


def _section_step_type(section: str) -> str:
    key = (section or "").strip().replace("’", "'").replace("`", "'")
    key = re.sub(r"\s+", " ", key).strip().lower()
    return _PLAN_SECTION_TYPES.get(key, "step")


def _counts_for_coverage(step: PlanStep) -> bool:
    return step.step_type not in _NON_EXECUTABLE_STEP_TYPES


def _looks_like_rich_plan_prompt(text: str) -> bool:
    cleaned = _strip_system_reminders(text or "")
    if not cleaned:
        return False
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    section_hits = sum(1 for line in lines if _canonical_section_name(line))
    numbered_hits = sum(1 for line in lines if re.match(r"^\d+[.)]\s+", line))
    bullet_hits = sum(1 for line in lines if re.match(r"^[-*]\s+", line))
    if section_hits >= 1 and (numbered_hits >= 1 or bullet_hits >= 3):
        return True
    return section_hits >= 2 or numbered_hits >= 3


def _derive_task_from_plan_text(plan: str, steps: list[PlanStep]) -> str:
    for preferred in ("required_step", "benchmark", "acceptance_gate", "assumption", "step"):
        for step in steps:
            if step.step_type == preferred and step.description.strip():
                return step.description.strip()[:240]
    cleaned = _strip_system_reminders(plan or "")
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    return first_line[:240]


def parse_plan_steps(plan: str) -> list[PlanStep]:
    text = _strip_system_reminders(plan or "")
    if not text:
        return []

    steps: list[PlanStep] = []
    current: list[str] = []
    current_id = 0
    current_section = ""
    bullet_re = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)\s*$")
    numbered_re = re.compile(r"^\s*(\d+[.)])\s+(.*\S)\s*$")

    def flush() -> None:
        nonlocal current, current_id
        if not current:
            return
        current_id += 1
        description = current[0].strip()
        acceptance = [line.strip() for line in current[1:] if line.strip()]
        steps.append(
            PlanStep(
                step_id=f"step-{current_id}",
                description=description,
                section=current_section,
                step_type=_section_step_type(current_section),
                acceptance_criteria=acceptance,
            )
        )
        current = []

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not line.strip():
            flush()
            continue
        section_name = _canonical_section_name(stripped)
        if section_name:
            flush()
            current_section = section_name
            continue
        m = bullet_re.match(line)
        if m:
            if numbered_re.match(line):
                flush()
                current = [m.group(1).strip()]
            elif _section_step_type(current_section) in {"evidence", "gap"}:
                flush()
                current = [m.group(1).strip()]
            elif current:
                current.append(m.group(1).strip())
            else:
                current = [m.group(1).strip()]
            continue
        if current:
            current.append(stripped)
        else:
            current = [stripped]
    flush()

    if steps:
        return steps

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[PlanStep] = []
    for idx, para in enumerate(paras, start=1):
        out.append(PlanStep(step_id=f"step-{idx}", description=para))
    return out


def build_change_summary(review_input: PlanReviewInput) -> str:
    changed_files = [str(p).strip() for p in review_input.changed_files if str(p).strip()]
    lines = []
    if review_input.task.strip():
        lines.append(f"Task: {review_input.task.strip()}")
    if changed_files:
        lines.append("Changed files:")
        for path in changed_files[:20]:
            lines.append(f"- {path}")
    if review_input.change_summary.strip():
        lines.append("Change summary:")
        lines.append(review_input.change_summary.strip())
    if review_input.execution_summary.strip():
        lines.append("Execution summary:")
        lines.append(review_input.execution_summary.strip())
    if review_input.diff_text.strip():
        diff = review_input.diff_text.strip()
        if len(diff) > 6000:
            diff = diff[:6000].rstrip() + "\n... [truncated diff]"
        lines.append("Diff excerpt:")
        lines.append(diff)
    return "\n".join(lines).strip()


def _status_from_string(value: Any) -> PlanStepStatus:
    raw = str(value or "").strip().lower()
    for status in PlanStepStatus:
        if status.value == raw:
            return status
    return PlanStepStatus.UNVERIFIED


def _extract_changed_files_from_diff(diff_text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"^\+\+\+\s+b/(.+)$", diff_text or "", flags=re.MULTILINE):
        path = match.group(1).strip()
        if path != "/dev/null" and path not in out:
            out.append(path)
    return out


class PlanReviewer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _build_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "yams_binary": self.config.yams_binary,
            "yams_data_dir": self.config.yams_data_dir,
        }
        if self.config.yams_cwd:
            kwargs["cwd"] = self.config.yams_cwd
        return kwargs

    def _normalize_input(self, review_input: PlanReviewInput) -> PlanReviewInput:
        plan_text = _strip_system_reminders(review_input.plan)
        task_text = str(review_input.task or "")
        if not plan_text and _looks_like_rich_plan_prompt(task_text):
            plan_text = _strip_system_reminders(task_text)
            task_text = _derive_task_from_plan_text(plan_text, parse_plan_steps(plan_text))
        files = [str(p).strip() for p in review_input.changed_files if str(p).strip()]
        if not files and review_input.diff_text:
            files = _extract_changed_files_from_diff(review_input.diff_text)
        max_files = max(1, int(self.config.plan_review_max_changed_files or 8))
        return PlanReviewInput(
            plan=plan_text,
            task=task_text,
            diff_text=review_input.diff_text,
            change_summary=review_input.change_summary,
            execution_summary=review_input.execution_summary,
            changed_files=files[:max_files],
        )

    def _build_plan_normalization_prompt(self, plan: str) -> list[dict[str, Any]]:
        schema = {
            "task_summary": "string",
            "steps": [
                {
                    "section": "string",
                    "step_type": "assumption|required_step|benchmark|acceptance_gate|decision_branch|evidence|gap|step",
                    "description": "string",
                    "acceptance_criteria": ["string"],
                }
            ],
        }
        sys = (
            "You normalize implementation plans into structured execution steps. "
            "Return JSON only. Preserve intent, but remove control metadata and wrapper text."
        )
        user = (
            "Normalize this plan-like prompt into execution-ready steps.\n\n"
            "Rules:\n"
            "- Keep assumptions as assumption steps.\n"
            "- Keep acceptance gates as acceptance_gate steps.\n"
            "- Keep completed-work/status sections as evidence steps, not executable requirements.\n"
            "- Keep blocker/risk sections as gap steps, not completed work.\n"
            "- Keep conditional future decisions as decision_branch steps.\n"
            "- For numbered sections, keep one top-level step per numbered item and fold nested bullets into acceptance_criteria.\n"
            "- Ignore <system-reminder> content if present.\n\n"
            f"Return JSON with schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"PLAN:\n{_strip_system_reminders(plan)}"
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    def _parse_normalized_plan_response(self, data: dict[str, Any]) -> tuple[str, list[PlanStep]]:
        task_summary = str(data.get("task_summary") or "").strip()
        raw_steps = data.get("steps") or []
        steps: list[PlanStep] = []
        if isinstance(raw_steps, list):
            for idx, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    continue
                desc = str(item.get("description") or "").strip()
                if not desc:
                    continue
                section = str(item.get("section") or "").strip()
                step_type = (
                    str(item.get("step_type") or _section_step_type(section)).strip() or "step"
                )
                criteria = item.get("acceptance_criteria") or []
                if not isinstance(criteria, list):
                    criteria = []
                steps.append(
                    PlanStep(
                        step_id=f"step-{idx}",
                        description=desc,
                        section=section,
                        step_type=step_type,
                        acceptance_criteria=[str(x).strip() for x in criteria if str(x).strip()],
                    )
                )
        return task_summary, steps

    async def _normalize_plan_with_model(
        self, review_input: PlanReviewInput, fallback_steps: list[PlanStep]
    ) -> tuple[str, list[PlanStep]]:
        if not _looks_like_rich_plan_prompt(review_input.plan):
            return review_input.task, fallback_steps

        critic_cfg: ModelConfig = self.config.critic_model or self.config.executor_model
        executor = ModelExecutor(critic_cfg)
        messages = self._build_plan_normalization_prompt(review_input.plan)
        try:
            exec_result = await executor.execute_raw(
                messages,
                model=critic_cfg.name,
                temperature=0.0,
                max_tokens=min(int(critic_cfg.max_output_tokens or 1024), 1600),
            )
            raw_output = exec_result.output or ""
            blob = _extract_first_json_object(raw_output or "")
            parsed = _try_parse_json(blob or raw_output or "")
            if isinstance(parsed, dict):
                task_summary, steps = self._parse_normalized_plan_response(parsed)
                if steps:
                    return task_summary or review_input.task, steps
        except Exception as e:
            logger.debug("Plan normalization model call failed: %s", e)
        return review_input.task, fallback_steps

    def _build_queries(
        self, task: str, steps: list[PlanStep], changed_files: list[str]
    ) -> list[QuerySpec]:
        specs: list[QuerySpec] = []
        seen: set[tuple[str, str]] = set()

        def add(query: str, query_type: QueryType, importance: float, reason: str) -> None:
            q = (query or "").strip()
            if not q:
                return
            key = (query_type.value, q)
            if key in seen:
                return
            seen.add(key)
            specs.append(
                QuerySpec(
                    query=q,
                    query_type=query_type,
                    importance=importance,
                    reason=reason,
                )
            )

        if task.strip():
            add(task.strip(), QueryType.SEMANTIC, 1.0, "review task context")

        for step in steps[:5]:
            add(step.description, QueryType.SEMANTIC, 0.9, "plan step retrieval")

        for path in changed_files[: int(self.config.plan_review_max_changed_files or 8)]:
            add(path, QueryType.GET, 1.0, "changed file inspection")
            add(path, QueryType.GRAPH, 0.7, "changed file graph context")
            name = Path(path).name
            if name:
                add(f"{name} path:{path}", QueryType.GREP, 0.6, "changed file grep anchor")

        return specs[: max(4, int(self.config.max_queries_per_iteration or 5) + 4)]

    async def _ingest_artifacts(
        self,
        client: YAMSClient,
        review_input: PlanReviewInput,
        steps: list[PlanStep],
        change_summary: str,
    ) -> list[str]:
        artifacts: list[str] = []
        task_slug = re.sub(r"[^a-z0-9]+", "-", (review_input.task or "plan-review").lower()).strip(
            "-"
        )
        task_slug = task_slug or "plan-review"
        base_meta = {
            "task": task_slug[:64],
            "phase": "checkpoint",
            "owner": "opencode",
            "mode": "engineering",
            "agent_id": "opencode-dcs-plan-review-benchmark",
            "status": "open",
        }
        try:
            plan_hash = await client.add(
                content=json.dumps(
                    {"task": review_input.task, "steps": [asdict(s) for s in steps]}, indent=2
                ),
                name=f"dcs-plan-review-plan-{task_slug}",
                tags=["dcs", "plan-review", "plan"],
                metadata={**base_meta, "source": "decision"},
            )
            artifacts.append(plan_hash)
        except Exception as e:
            logger.debug("Plan review plan ingestion failed: %s", e)

        try:
            change_hash = await client.add(
                content=change_summary,
                name=f"dcs-plan-review-changes-{task_slug}",
                tags=["dcs", "plan-review", "changes"],
                metadata={**base_meta, "source": "evidence"},
            )
            artifacts.append(change_hash)
        except Exception as e:
            logger.debug("Plan review change ingestion failed: %s", e)

        return artifacts

    def _review_prompt(
        self,
        review_input: PlanReviewInput,
        context: ContextBlock,
        steps: list[PlanStep],
        change_summary: str,
    ) -> list[dict[str, Any]]:
        schema = {
            "coverage_score": "number 0.0-1.0",
            "executed_well": "boolean",
            "summary": "string",
            "gaps": ["string"],
            "advice": ["string"],
            "suggested_tests": ["string"],
            "followup_plan": ["string"],
            "step_reviews": [
                {
                    "step_id": "string",
                    "description": "string",
                    "status": "complete|partial|missing|unverified",
                    "confidence": "number 0.0-1.0",
                    "evidence": ["string"],
                    "gaps": ["string"],
                }
            ],
        }
        sys = (
            "You are a strict code-change plan reviewer. Judge whether the implementation changes "
            "satisfied the plan using only the supplied change summary and retrieved repository context. "
            "Return JSON only. Be conservative when evidence is weak."
        )
        plan_block = json.dumps([asdict(s) for s in steps], indent=2)
        user = (
            f"TASK:\n{(review_input.task or '').strip()}\n\n"
            f"PLAN:\n{plan_block}\n\n"
            f"CHANGES:\n{change_summary}\n\n"
            f"RETRIEVED CONTEXT:\n{context.content}\n\n"
            f"Return JSON with this schema:\n{json.dumps(schema, indent=2)}"
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    def _heuristic_review(
        self,
        review_input: PlanReviewInput,
        steps: list[PlanStep],
        query_results: list[YAMSQueryResult],
        context: ContextBlock,
    ) -> PlanReviewResult:
        haystack = "\n".join(
            [
                review_input.change_summary,
                review_input.diff_text,
                review_input.execution_summary,
                context.content,
            ]
        ).lower()
        step_reviews: list[PlanStepReview] = []
        completed = 0
        coverage_steps = [step for step in steps if _counts_for_coverage(step)]
        gaps: list[str] = []
        for step in steps:
            tokens = [
                t.lower() for t in re.findall(r"[A-Za-z0-9_./-]+", step.description) if len(t) >= 4
            ]
            hits = sum(1 for token in tokens[:8] if token in haystack)
            if step.step_type == "evidence":
                status = PlanStepStatus.COMPLETE if hits > 0 else PlanStepStatus.UNVERIFIED
            elif step.step_type == "gap":
                status = PlanStepStatus.PARTIAL if hits > 0 else PlanStepStatus.MISSING
                if hits > 0:
                    gaps.append(f"Blocker still present: {step.description}")
                else:
                    gaps.append(f"Gap remains open: {step.description}")
            elif tokens and hits >= max(1, len(tokens[:4]) // 2):
                status = PlanStepStatus.COMPLETE
                completed += 1
            elif hits > 0:
                status = PlanStepStatus.PARTIAL
                gaps.append(f"Partial evidence for: {step.description}")
            else:
                status = PlanStepStatus.MISSING
                if _counts_for_coverage(step):
                    gaps.append(f"No clear evidence for: {step.description}")
            step_reviews.append(
                PlanStepReview(
                    step_id=step.step_id,
                    description=step.description,
                    status=status,
                    confidence=1.0 if status == PlanStepStatus.COMPLETE else (0.5 if hits else 0.1),
                    evidence=[src for src in review_input.changed_files[:3]],
                    gaps=[]
                    if status == PlanStepStatus.COMPLETE
                    else [f"Verify {step.description}"],
                )
            )

        coverage = completed / max(1, len(coverage_steps)) if coverage_steps else 0.0
        sources: list[str] = []
        for qr in query_results:
            for chunk in qr.chunks:
                if chunk.source and chunk.source not in sources:
                    sources.append(chunk.source)
        advice = ["Add targeted verification for any partially implemented steps."] if gaps else []
        tests: list[str] = []
        for step in steps:
            if step.step_type in {"required_step", "acceptance_gate", "step"} and step.description:
                tests.append(f"Add regression test for {step.description}")
            elif step.step_type == "benchmark" and step.description:
                tests.append(f"Rerun benchmark case: {step.description}")
            if len(tests) >= 3:
                break
        followup: list[str] = []
        for step in steps:
            if step.step_type == "gap" and step.description:
                followup.append(step.description)
        for gap in gaps:
            if gap not in followup:
                followup.append(gap)
        followup = followup[:3]
        return PlanReviewResult(
            task=review_input.task or "",
            plan_steps=steps,
            step_reviews=step_reviews,
            coverage_score=coverage,
            executed_well=coverage >= 0.8 and not gaps,
            summary="Heuristic review completed.",
            gaps=gaps,
            advice=advice,
            suggested_tests=tests,
            followup_plan=followup,
            changed_files=list(review_input.changed_files),
            retrieved_sources=sources,
            context=context,
            query_results=query_results,
        )

    def _parse_review_response(
        self,
        data: dict[str, Any],
        review_input: PlanReviewInput,
        steps: list[PlanStep],
        query_results: list[YAMSQueryResult],
        context: ContextBlock,
        raw_output: str,
        latency_ms: float,
        ingested_artifacts: list[str],
    ) -> PlanReviewResult:
        reviews_raw = data.get("step_reviews")
        reviews: list[PlanStepReview] = []
        if isinstance(reviews_raw, list):
            for item in reviews_raw:
                if not isinstance(item, dict):
                    continue
                reviews.append(
                    PlanStepReview(
                        step_id=str(item.get("step_id") or ""),
                        description=str(item.get("description") or ""),
                        status=_status_from_string(item.get("status")),
                        confidence=_clamp01(item.get("confidence") or 0.0),
                        evidence=[str(x) for x in (item.get("evidence") or []) if x is not None],
                        gaps=[str(x) for x in (item.get("gaps") or []) if x is not None],
                    )
                )

        if not reviews:
            reviews = [
                PlanStepReview(
                    step_id=step.step_id,
                    description=step.description,
                    status=PlanStepStatus.UNVERIFIED,
                )
                for step in steps
            ]

        sources: list[str] = []
        for qr in query_results:
            for chunk in qr.chunks:
                if chunk.source and chunk.source not in sources:
                    sources.append(chunk.source)

        return PlanReviewResult(
            task=review_input.task or "",
            plan_steps=steps,
            step_reviews=reviews,
            coverage_score=_clamp01(data.get("coverage_score") or 0.0),
            executed_well=bool(data.get("executed_well", False)),
            summary=str(data.get("summary") or "").strip(),
            gaps=[str(x) for x in (data.get("gaps") or []) if x is not None],
            advice=[str(x) for x in (data.get("advice") or []) if x is not None],
            suggested_tests=[str(x) for x in (data.get("suggested_tests") or []) if x is not None],
            followup_plan=[str(x) for x in (data.get("followup_plan") or []) if x is not None],
            changed_files=list(review_input.changed_files),
            retrieved_sources=sources,
            ingested_artifacts=list(ingested_artifacts),
            context=context,
            query_results=query_results,
            model_output=raw_output,
            latency_ms=latency_ms,
            raw_response=data,
        )

    async def review(self, review_input: PlanReviewInput) -> PlanReviewResult:
        start = time.perf_counter()
        normalized = self._normalize_input(review_input)
        steps = parse_plan_steps(normalized.plan)
        normalized_task, steps = await self._normalize_plan_with_model(normalized, steps)
        if normalized_task:
            normalized.task = normalized_task
        change_summary = build_change_summary(normalized)
        query_results: list[YAMSQueryResult] = []
        context = ContextBlock(content="", budget=self.config.plan_review_context_budget)
        ingested_artifacts: list[str] = []

        async with YAMSClient(**self._build_client_kwargs()) as client:
            ingested_artifacts = await self._ingest_artifacts(
                client, normalized, steps, change_summary
            )
            specs = self._build_queries(normalized.task, steps, normalized.changed_files)
            for spec in specs:
                try:
                    result = await client.execute_spec(spec)
                except Exception as e:
                    result = YAMSQueryResult(spec=spec, chunks=[], error=str(e))
                result.chunks = list(result.chunks or [])[
                    : int(self.config.plan_review_search_limit or 4)
                ]
                query_results.append(result)

            assembler = ContextAssembler(
                budget=int(self.config.plan_review_context_budget or 1536),
                model=self.config.executor_model.name,
            )

            synthetic = list(query_results)
            if change_summary:
                synthetic.append(
                    YAMSQueryResult(
                        spec=QuerySpec(
                            query="plan-review-change-summary",
                            query_type=QueryType.GET,
                            importance=1.0,
                            reason="local change summary",
                        ),
                        chunks=[
                            YAMSChunk(
                                chunk_id="plan-review-change-summary",
                                content=change_summary,
                                score=1.0,
                                source="plan-review",
                            )
                        ],
                    )
                )
            context = assembler.assemble(synthetic, task=normalized.task)

        critic_cfg: ModelConfig = self.config.critic_model or self.config.executor_model
        executor = ModelExecutor(critic_cfg)
        messages = self._review_prompt(normalized, context, steps, change_summary)
        raw_output = ""
        data: dict[str, Any] | None = None
        try:
            exec_result = await executor.execute_raw(
                messages,
                model=critic_cfg.name,
                temperature=min(0.2, float(critic_cfg.temperature or 0.0)),
                max_tokens=min(int(critic_cfg.max_output_tokens or 1024), 1200),
            )
            raw_output = exec_result.output or ""
            blob = _extract_first_json_object(raw_output or "")
            parsed = _try_parse_json(blob or raw_output or "")
            if isinstance(parsed, dict):
                data = parsed
        except Exception as e:
            logger.warning("Plan review model call failed: %s", e)

        latency_ms = (time.perf_counter() - start) * 1000.0
        if data is None:
            result = self._heuristic_review(normalized, steps, query_results, context)
            result.ingested_artifacts = list(ingested_artifacts)
            result.model_output = raw_output
            result.latency_ms = latency_ms
            return result
        return self._parse_review_response(
            data,
            normalized,
            steps,
            query_results,
            context,
            raw_output,
            latency_ms,
            ingested_artifacts,
        )


__all__ = [
    "PlanReviewer",
    "PlanReviewInput",
    "PlanReviewResult",
    "PlanStep",
    "PlanStepReview",
    "PlanStepStatus",
    "build_change_summary",
    "parse_plan_steps",
]
