from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any

try:  # optional dependency
    import dspy  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    dspy = None  # type: ignore[assignment]

from rich.console import Console
from rich.table import Table

from dcs.assembler import ContextAssembler
from dcs.client import YAMSClient
from dcs.codemap import CodemapBuilder
from dcs.critic import SelfCritic
from dcs.decomposer import TaskDecomposer
from dcs.executor import ModelExecutor
from dcs.faithfulness import build_abstention_output, build_faithfulness_report
from dcs.lmstudio_context import (
    count_prompt_tokens,
    get_context_length,
)
from dcs.lmstudio_context import (
    is_available as lmstudio_available,
)
from dcs.optimizer import RetrievalOptimizer
from dcs.planner import QueryPlanner
from dcs.types import (
    ContextBlock,
    Critique,
    ExecutionResult,
    IterationRecord,
    PipelineConfig,
    PipelineResult,
    QuerySpec,
    QueryType,
    YAMSQueryResult,
)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _is_lmstudio_backend(base_url: str) -> bool:
    raw = str(base_url or "").lower()
    return "127.0.0.1:8080" not in raw and "/api/v1" not in raw and "localhost:1234" in raw


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _merge_weights(base: dict[str, float], update: dict[str, float]) -> dict[str, float]:
    merged = dict(base or {})
    for k, v in (update or {}).items():
        try:
            merged[str(k)] = float(v)
        except Exception:
            continue
    return merged


class DCSPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()

    def _print_iteration_table(self, record: IterationRecord) -> None:
        t = Table(title=f"Iteration {record.iteration}")
        t.add_column("Specs", justify="right")
        t.add_column("Chunks", justify="right")
        t.add_column("Ctx tokens", justify="right")
        t.add_column("Ctx util", justify="right")
        t.add_column("Exec ms", justify="right")
        t.add_column("Quality", justify="right")
        t.add_column("Missing", justify="right")
        t.add_column("Irrelevant", justify="right")

        specs_n = len(record.specs)
        chunks_considered = record.context.chunks_considered if record.context else 0
        chunks_included = record.context.chunks_included if record.context else 0
        chunks_text = f"{chunks_included}/{chunks_considered}"
        ctx_tokens = record.context.token_count if record.context else 0
        ctx_util = record.context.utilization if record.context else 0.0
        exec_ms = record.result.latency_ms if record.result else 0.0
        quality = record.critique.quality_score if record.critique else 0.0
        missing = len(record.critique.missing_info) if record.critique else 0
        irrelevant = len(record.critique.irrelevant_chunks) if record.critique else 0

        t.add_row(
            str(specs_n),
            chunks_text,
            str(ctx_tokens),
            f"{ctx_util:.2f}",
            f"{exec_ms:.0f}",
            f"{quality:.2f}",
            str(missing),
            str(irrelevant),
        )
        self.console.print(t)

    def _converged(self, prev_quality: float | None, critique: Critique) -> bool:
        q = float(critique.quality_score or 0.0)
        if q >= float(self.config.quality_threshold):
            return True
        if prev_quality is None:
            return False
        return abs(q - float(prev_quality)) < float(self.config.convergence_delta)

    def _build_client_kwargs(self, weights: dict[str, float]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        # These are best-effort; client implementation may accept or ignore.
        kwargs["yams_binary"] = self.config.yams_binary
        kwargs["yams_data_dir"] = self.config.yams_data_dir
        kwargs["search_weights"] = weights
        kwargs["request_timeout_s"] = max(
            30.0, float(self.config.executor_model.request_timeout_s or 60.0)
        )
        if self.config.yams_cwd:
            kwargs["cwd"] = self.config.yams_cwd
        return kwargs

    def _reconcile_model_context_window(self, model_cfg) -> tuple[int, int]:
        requested = int(getattr(model_cfg, "context_window", 0) or 0)
        actual = requested
        if lmstudio_available() and _is_lmstudio_backend(getattr(model_cfg, "base_url", "")):
            try:
                detected = get_context_length(str(model_cfg.name or ""))
                if detected and int(detected) > 0:
                    actual = int(detected)
                    model_cfg.context_window = int(detected)
            except Exception:
                pass
        return requested, actual

    def _apply_context_profile(self, executor_ctx_window: int) -> None:
        profile = str(getattr(self.config, "context_profile", "auto") or "auto").lower()
        apply_large = profile == "large" or (
            profile == "auto" and executor_ctx_window >= int(self.config.large_context_threshold)
        )
        if not apply_large:
            return

        changed: list[str] = []
        if int(self.config.context_budget) == 2048:
            self.config.context_budget = int(self.config.large_context_budget)
            changed.append(f"context_budget={self.config.context_budget}")
        if int(self.config.system_prompt_budget) == 512:
            self.config.system_prompt_budget = int(self.config.large_system_prompt_budget)
            changed.append(f"system_prompt_budget={self.config.system_prompt_budget}")
        if int(self.config.output_reserve) == 1024:
            self.config.output_reserve = int(self.config.large_output_reserve)
            changed.append(f"output_reserve={self.config.output_reserve}")
        if int(self.config.codemap_budget) == 256:
            self.config.codemap_budget = int(self.config.large_codemap_budget)
            changed.append(f"codemap_budget={self.config.codemap_budget}")

        if changed:
            self.console.print(
                "[dim]Context profile=large applied: " + ", ".join(changed) + "[/dim]"
            )

    def _build_dspy_retrieval_model(self) -> Any | None:
        if not self.config.use_dspy_retrieval_rerank or dspy is None:
            return None
        model_cfg = (
            self.config.dspy_retrieval_model
            or self.config.critic_model
            or self.config.executor_model
        )
        candidates = [model_cfg.name]
        if not model_cfg.name.startswith("openai/"):
            candidates.append(f"openai/{model_cfg.name}")
        last_err: Exception | None = None
        for model_name in candidates:
            try:
                return dspy.LM(
                    model_name,
                    api_base=model_cfg.base_url,
                    api_key=model_cfg.api_key,
                    temperature=0.0,
                    max_tokens=max(128, int(self.config.dspy_retrieval_max_tokens or 16384)),
                    timeout=float(model_cfg.request_timeout_s),
                )
            except Exception as e:  # pragma: no cover
                last_err = e
                continue
        if last_err is not None:
            self.console.print(f"[dim]DSPy retrieval rerank unavailable: {last_err}[/dim]")
        return None

    def _prepend_codemap(
        self,
        context: ContextBlock,
        codemap_prefix: str,
        codemap_tokens: int,
    ) -> ContextBlock:
        if not codemap_prefix:
            return context
        combined_content = codemap_prefix.rstrip() + "\n\n" + context.content
        combined_tokens = context.token_count + codemap_tokens
        return ContextBlock(
            content=combined_content,
            sources=["yams-knowledge-graph"] + list(context.sources or []),
            chunk_ids=["codemap"] + list(context.chunk_ids or []),
            token_count=combined_tokens,
            budget=self.config.context_budget,
            utilization=combined_tokens / max(1, self.config.context_budget),
            chunks_included=context.chunks_included + 1,
            chunks_considered=context.chunks_considered,
        )

    def _compute_decomposer_controls(
        self,
        optimizer: RetrievalOptimizer,
        prev_critique: Critique | None,
    ) -> tuple[dict[str, float], set[QueryType]]:
        scores = optimizer.get_query_type_scores()
        type_bias: dict[str, float] = {
            QueryType.SEMANTIC.value: 1.0,
            QueryType.GREP.value: 1.0,
            QueryType.GET.value: 1.0,
            QueryType.GRAPH.value: 1.0,
            QueryType.LIST.value: 1.0,
        }
        require_types: set[QueryType] = set()

        semantic_score = float(scores.get("semantic", 0.5))
        grep_score = float(scores.get("grep", 0.5))
        get_score = float(scores.get("get", 0.5))

        if semantic_score < 0.45:
            type_bias[QueryType.SEMANTIC.value] *= 0.75
            type_bias[QueryType.GREP.value] *= 1.25
            type_bias[QueryType.GET.value] *= 1.15
            require_types.add(QueryType.GREP)

        if get_score < 0.35 and grep_score < 0.45:
            # Keep at least one semantic probe when lexical lookups also struggle.
            type_bias[QueryType.SEMANTIC.value] *= 1.1

        if prev_critique and len(prev_critique.missing_info or []) >= 2:
            type_bias[QueryType.GET.value] *= 1.2
            require_types.add(QueryType.GET)

        return type_bias, require_types

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
            "explain",
            "describe",
        }
        terms: list[str] = []
        for w in re.findall(r"[A-Za-z0-9_\-]+", task or ""):
            t = w.lower()
            if len(t) < 4 or t in stop:
                continue
            if t not in terms:
                terms.append(t)
            if len(terms) >= 12:
                break
        return terms

    @staticmethod
    def _query_terms(spec: QuerySpec) -> list[str]:
        terms: list[str] = []
        for w in re.findall(r"[A-Za-z0-9_\-]+", spec.query or ""):
            if w.lower() == "path":
                continue
            if len(w) < 3:
                continue
            lw = w.lower()
            if lw not in terms:
                terms.append(lw)
        return terms

    @staticmethod
    def _spec_has_path_hint(spec: QuerySpec) -> bool:
        q = (spec.query or "").lower()
        return "path:" in q or "/" in q

    @staticmethod
    def _is_test_source(src: str) -> bool:
        s = (src or "").lower()
        return "/tests/" in s or s.endswith("_test.cpp") or s.endswith("_test.py")

    def _rerank_and_cap_chunks(
        self,
        *,
        task: str,
        spec: QuerySpec,
        chunks: list[Any],
    ) -> list[Any]:
        if not chunks:
            return []

        task_l = (task or "").lower()
        task_terms = self._extract_task_terms(task)
        query_terms = self._query_terms(spec)
        has_path_hint = self._spec_has_path_hint(spec)
        mentions_test = "test" in task_l

        rescored: list[Any] = []
        for c in list(chunks):
            score = float(getattr(c, "score", 0.0) or 0.0)
            src_l = (getattr(c, "source", "") or "").lower()
            text_l = (getattr(c, "content", "") or "").lower()

            # Penalize obvious noisy paths unless task asks for tests.
            if not mentions_test and self._is_test_source(src_l):
                score *= 0.6

            if "/docs/" in src_l and "doc" not in task_l:
                score *= 0.85

            # Reward source paths that overlap task terms; downweight otherwise.
            if task_terms:
                path_hits = sum(1 for t in task_terms if t in src_l)
                if path_hits > 0:
                    score *= 1.0 + min(0.2, 0.05 * path_hits)
                elif spec.query_type == QueryType.GREP and not has_path_hint:
                    score *= 0.82

            # Reward chunks that mention query terms in content.
            if query_terms:
                qhits = sum(1 for t in query_terms if t in text_l)
                if qhits > 0:
                    score *= 1.0 + min(0.2, 0.05 * qhits)
                elif spec.query_type == QueryType.GREP:
                    score *= 0.9

            c.score = max(0.0, min(1.0, score))
            rescored.append(c)

        rescored = [
            c
            for c in rescored
            if float(getattr(c, "score", 0.0) or 0.0) >= float(self.config.min_chunk_score)
        ]

        rescored.sort(key=lambda c: float(getattr(c, "score", 0.0) or 0.0), reverse=True)

        # Keep non-test files first when available.
        if spec.query_type == QueryType.GREP and not mentions_test:
            non_test = [c for c in rescored if not self._is_test_source(getattr(c, "source", ""))]
            if non_test:
                rescored = non_test

        # Avoid over-concentrating on one file for broad grep.
        if spec.query_type == QueryType.GREP and not has_path_hint:
            per_source: dict[str, int] = {}
            balanced: list[Any] = []
            for c in rescored:
                src = str(getattr(c, "source", "") or getattr(c, "chunk_id", ""))
                n = per_source.get(src, 0)
                if n >= 2:
                    continue
                per_source[src] = n + 1
                balanced.append(c)
            rescored = balanced

        max_chunks = int(self.config.max_chunks_per_query)
        if spec.query_type == QueryType.GREP:
            max_chunks = min(max_chunks, 6)
        elif spec.query_type == QueryType.GET:
            max_chunks = min(max_chunks, 4)

        return rescored[:max_chunks]

    async def _execute_with_overflow_retry(
        self,
        *,
        executor: ModelExecutor,
        task: str,
        query_results: list[YAMSQueryResult],
        context: ContextBlock,
        assembler: ContextAssembler,
        codemap_prefix: str,
        codemap_tokens: int,
    ) -> tuple[ExecutionResult, ContextBlock, ContextAssembler, int]:
        retries = 0
        cur_context = context
        cur_assembler = assembler
        cur_max_tokens = int(self.config.executor_model.max_output_tokens)

        while True:
            exec_result = await executor.execute(
                task=task,
                context=cur_context,
                max_tokens=cur_max_tokens,
            )
            err = (
                (exec_result.raw_response or {}).get("error") if exec_result.raw_response else None
            )
            if err != "context_overflow":
                return exec_result, cur_context, cur_assembler, retries

            if retries >= int(self.config.max_context_overflow_retries):
                return exec_result, cur_context, cur_assembler, retries

            retries += 1
            shrink = _clamp01(float(self.config.context_shrink_factor))
            if shrink <= 0.0:
                shrink = 0.7

            next_budget = max(
                int(self.config.min_context_budget),
                int(cur_assembler.budget * shrink),
            )
            next_max_tokens = max(
                int(self.config.min_output_tokens),
                int(cur_max_tokens * shrink),
            )

            if next_budget >= cur_assembler.budget and next_max_tokens >= cur_max_tokens:
                return exec_result, cur_context, cur_assembler, retries

            cur_max_tokens = next_max_tokens
            cur_assembler = ContextAssembler(
                budget=next_budget,
                model=self.config.executor_model.name,
            )
            rebuilt = cur_assembler.assemble(query_results, task=task)
            cur_context = self._prepend_codemap(rebuilt, codemap_prefix, codemap_tokens)

    def _init_client(self, weights: dict[str, float]) -> YAMSClient:
        """Construct YAMSClient with best-effort config injection.

        The interface spec for YAMSClient in this project doesn't guarantee an __init__
        signature, so we attempt kwargs then fall back to attribute assignment.
        """

        kwargs = self._build_client_kwargs(weights)
        try:
            return YAMSClient(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            c = YAMSClient()  # type: ignore[call-arg]
            for k, v in kwargs.items():
                if hasattr(c, k):
                    try:
                        setattr(c, k, v)
                    except Exception:
                        pass
            return c

    async def run(self, task: str) -> PipelineResult:
        start_total = _now_ms()
        result = PipelineResult(task=task)

        critic_cfg = self.config.critic_model or self.config.executor_model
        base_weights = dict(self.config.search_weights or {})

        # Runtime context-size check: requested vs actual (LM Studio).
        req_exec, act_exec = self._reconcile_model_context_window(self.config.executor_model)
        req_crit, act_crit = self._reconcile_model_context_window(critic_cfg)
        self.console.print(
            f"[dim]Context window executor: requested={req_exec} actual={act_exec}[/dim]"
        )
        if critic_cfg.name != self.config.executor_model.name or act_crit != act_exec:
            self.console.print(
                f"[dim]Context window critic: requested={req_crit} actual={act_crit}[/dim]"
            )

        self._apply_context_profile(int(act_exec or req_exec or 0))

        self.console.rule("DCS Pipeline")
        self.console.print(f"Task: {task}")
        self.console.print(
            f"Executor: {self.config.executor_model.name} | Critic: {critic_cfg.name} | "
            f"Budget: {self.config.context_budget} | Iterations: {self.config.max_iterations}"
        )

        prev_specs: list[QuerySpec] = []
        prev_critique: Critique | None = None
        prev_quality: float | None = None

        try:
            weights = dict(base_weights)
            async with self._init_client(weights) as client:
                executor = ModelExecutor(self.config.executor_model)
                decomposer = TaskDecomposer(self.config.executor_model)
                dspy_retrieval_model = self._build_dspy_retrieval_model()
                planner = QueryPlanner(
                    client,
                    max_concurrency=int(self.config.retrieval_max_concurrency),
                    dspy_rerank_model=dspy_retrieval_model,
                    dspy_rerank_top_k=int(self.config.dspy_retrieval_top_k),
                    dspy_rerank_prefer_json=bool(self.config.dspy_retrieval_prefer_json),
                )
                assembler = ContextAssembler(
                    budget=self.config.context_budget, model=self.config.executor_model.name
                )
                critic = SelfCritic(critic_cfg)
                optimizer = RetrievalOptimizer(yams_client=client)

                # Build structural codemap from YAMS knowledge graph (once, before iterations)
                codemap_prefix = ""
                codemap_tokens = 0
                if self.config.codemap_budget > 0:
                    try:
                        t0 = _now_ms()
                        codemap_builder = CodemapBuilder(
                            client,
                            token_budget=self.config.codemap_budget,
                            max_files=max(1, int(self.config.codemap_max_files)),
                            max_symbols_per_file=max(
                                1, int(self.config.codemap_max_symbols_per_file)
                            ),
                            include_type_counts=bool(self.config.codemap_include_type_counts),
                        )
                        codemap_result = await codemap_builder.build(task=task)
                        codemap_prefix = codemap_result.tree_text
                        codemap_tokens = codemap_result.context_block.token_count
                        self.console.print(
                            f"[dim]Codemap: {codemap_result.node_count} nodes, "
                            f"{codemap_tokens} tokens, {codemap_result.latency_ms:.0f}ms[/dim]"
                        )
                    except Exception as e:
                        self.console.print(f"[dim]Codemap: skipped ({e})[/dim]")

                # Adjust assembler budget to reserve room for codemap prefix and
                # keep total prompt within the model context window.
                effective_budget = int(self.config.context_budget)
                model_budget = ContextAssembler.estimate_budget(
                    self.config.executor_model.context_window,
                    self.config.system_prompt_budget,
                    self.config.output_reserve,
                )
                if model_budget > 0:
                    effective_budget = min(effective_budget, model_budget)

                if codemap_tokens > 0:
                    effective_budget = max(0, effective_budget - codemap_tokens)

                if effective_budget <= 0:
                    self.console.print(
                        "[dim]Assembler budget reduced to 0 (context window too small)[/dim]"
                    )
                else:
                    assembler = ContextAssembler(
                        budget=effective_budget, model=self.config.executor_model.name
                    )
                    if effective_budget != self.config.context_budget or codemap_tokens > 0:
                        self.console.print(
                            f"[dim]Assembler budget adjusted: {self.config.context_budget} "
                            f"(model={model_budget}) - {codemap_tokens} codemap = {effective_budget}[/dim]"
                        )

                for it in range(1, int(self.config.max_iterations) + 1):
                    iter_start = _now_ms()
                    record = IterationRecord(iteration=it)
                    result.iterations.append(record)

                    stage_table = Table(title=f"Iteration {it} Stages")
                    stage_table.add_column("Stage")
                    stage_table.add_column("ms", justify="right")
                    stage_table.add_column("Details")

                    try:
                        type_bias, require_types = self._compute_decomposer_controls(
                            optimizer,
                            prev_critique,
                        )

                        # a) Decompose/refine
                        t0 = _now_ms()
                        if it == 1 or prev_critique is None:
                            specs = await decomposer.decompose(
                                task,
                                max_queries=self.config.max_queries_per_iteration,
                                type_bias=type_bias,
                                require_types=require_types,
                                use_task_seeding=bool(self.config.enable_task_seeding),
                            )
                        else:
                            specs = await decomposer.refine(
                                task,
                                prev_critique,
                                prev_specs,
                                type_bias=type_bias,
                                require_types=require_types,
                                use_task_seeding=bool(self.config.enable_task_seeding),
                            )
                        specs = list(specs or [])[: int(self.config.max_queries_per_iteration)]
                        record.specs = specs
                        stage_table.add_row(
                            "decompose" if it == 1 else "refine",
                            f"{_now_ms() - t0:.0f}",
                            f"specs={len(specs)}",
                        )

                        # b) Plan/retrieve
                        t0 = _now_ms()
                        query_results: list[YAMSQueryResult] = await planner.execute(specs)

                        # Enforce local caps/filters regardless of planner implementation.
                        for qr in query_results:
                            qr.chunks = self._rerank_and_cap_chunks(
                                task=task,
                                spec=qr.spec,
                                chunks=list(qr.chunks or []),
                            )

                        record.query_results = query_results
                        stage_table.add_row(
                            "retrieve",
                            f"{_now_ms() - t0:.0f}",
                            f"queries={len(query_results)}",
                        )

                        # c) Assemble
                        t0 = _now_ms()
                        context = assembler.assemble(query_results, task=task)
                        context = self._prepend_codemap(context, codemap_prefix, codemap_tokens)

                        # Ensure prompt fits the actual LM Studio context length when available.
                        if lmstudio_available() and _is_lmstudio_backend(
                            getattr(self.config.executor_model, "base_url", "")
                        ):
                            max_ctx = get_context_length(self.config.executor_model.name)
                            if max_ctx:
                                target = int(max_ctx * 0.9)
                                for _ in range(3):
                                    messages = executor._build_messages(
                                        task=task, context=context, system_prompt=None
                                    )
                                    prompt_tokens = count_prompt_tokens(
                                        self.config.executor_model.name, messages
                                    )
                                    if prompt_tokens is None or prompt_tokens <= target:
                                        break
                                    shrink = max(
                                        64, int(assembler.budget * (target / max(1, prompt_tokens)))
                                    )
                                    if shrink >= assembler.budget:
                                        break
                                    assembler = ContextAssembler(
                                        budget=shrink,
                                        model=self.config.executor_model.name,
                                    )
                                    context = assembler.assemble(query_results, task=task)
                                    context = self._prepend_codemap(
                                        context,
                                        codemap_prefix,
                                        codemap_tokens,
                                    )

                        stage_table.add_row(
                            "assemble",
                            f"{_now_ms() - t0:.0f}",
                            f"tokens={context.token_count} util={context.utilization:.2f}",
                        )

                        # d) Execute
                        t0 = _now_ms()
                        (
                            exec_result,
                            context,
                            assembler,
                            overflow_retries,
                        ) = await self._execute_with_overflow_retry(
                            executor=executor,
                            task=task,
                            query_results=query_results,
                            context=context,
                            assembler=assembler,
                            codemap_prefix=codemap_prefix,
                            codemap_tokens=codemap_tokens,
                        )
                        record.result = exec_result
                        record.context = context
                        stage_table.add_row(
                            "execute",
                            f"{_now_ms() - t0:.0f}",
                            f"model={exec_result.model} retries={overflow_retries}",
                        )

                        # e) Critique
                        t0 = _now_ms()
                        critique = await critic.critique(
                            task=task, context=context, result=exec_result
                        )
                        record.critique = critique
                        stage_table.add_row(
                            "critique",
                            f"{_now_ms() - t0:.0f}",
                            f"quality={critique.quality_score:.2f}",
                        )

                        # e.1) Ground-truth-free faithfulness checks
                        if bool(self.config.no_ground_truth_mode):
                            t0 = _now_ms()
                            faith = build_faithfulness_report(
                                task=task,
                                context=context,
                                output=exec_result.output,
                                min_overlap=float(self.config.claim_evidence_min_overlap),
                                min_confidence=float(self.config.faithfulness_min_confidence),
                                max_unsupported_ratio=float(
                                    self.config.faithfulness_max_unsupported_ratio
                                ),
                                min_supported_claims=int(
                                    self.config.faithfulness_min_supported_claims
                                ),
                                use_dspy=bool(self.config.use_dspy_faithfulness),
                                dspy_model_config=critic_cfg,
                            )
                            record.faithfulness = faith

                            # Blend confidence into quality so router/escalation can
                            # treat weakly-grounded answers conservatively.
                            critique.quality_score = min(
                                float(critique.quality_score or 0.0),
                                float(faith.confidence or 0.0),
                            )

                            if faith.should_abstain:
                                critique.missing_info.append(
                                    "Faithfulness guard: low grounded confidence; additional evidence required."
                                )
                                if it >= int(self.config.max_iterations):
                                    exec_result.output = build_abstention_output(task, faith)
                                    record.result = exec_result

                            stage_table.add_row(
                                "faithfulness",
                                f"{_now_ms() - t0:.0f}",
                                f"conf={faith.confidence:.2f} support={faith.supported_ratio:.2f}",
                            )

                        # f) Optimize
                        t0 = _now_ms()
                        try:
                            optimizer.record_feedback(
                                query_results=query_results, critique=critique
                            )
                            adjusted = optimizer.get_adjusted_weights() or {}
                            weights = _merge_weights(base_weights, adjusted)
                            # Best-effort: update client weights if it exposes an attribute.
                            if hasattr(client, "search_weights"):
                                client.search_weights = weights
                        except Exception as e:
                            stage_table.add_row(
                                "optimize",
                                f"{_now_ms() - t0:.0f}",
                                f"error={e}",
                            )
                        else:
                            stage_table.add_row(
                                "optimize",
                                f"{_now_ms() - t0:.0f}",
                                f"weights={len(weights)}",
                            )

                        record.latency_ms = _now_ms() - iter_start

                        self.console.print(stage_table)
                        self._print_iteration_table(record)

                        prev_specs = specs
                        prev_critique = critique

                        faith_abstain = bool(
                            record.faithfulness is not None and record.faithfulness.should_abstain
                        )
                        if self._converged(prev_quality, critique) and not faith_abstain:
                            result.converged = True
                            break
                        prev_quality = float(critique.quality_score or 0.0)

                    except Exception as e:
                        record.latency_ms = _now_ms() - iter_start
                        record.result = ExecutionResult(output=f"Error in iteration {it}: {e}")
                        self.console.print(stage_table)
                        self.console.print(f"Iteration {it} failed: {e}")
                        break

        except Exception as e:
            # Failed to even initialize pipeline dependencies.
            err_record = IterationRecord(
                iteration=1,
                specs=[],
                query_results=[],
                context=ContextBlock(content="", token_count=0, budget=self.config.context_budget),
                result=ExecutionResult(output=f"Pipeline initialization error: {e}"),
                critique=None,
                latency_ms=0.0,
            )
            result.iterations.append(err_record)

        # Finalize — pick the best iteration by quality score (not blindly last).
        best_iter: IterationRecord | None = None
        if result.iterations:
            # Prefer iteration with highest critique quality_score.
            scored = [
                it for it in result.iterations if it.critique is not None and it.result is not None
            ]
            if scored:
                best_iter = max(
                    scored,
                    key=lambda it: (
                        float((it.critique.quality_score if it.critique else 0.0) or 0.0),
                        int(
                            it.context.token_count if it.context else 0
                        ),  # tiebreak: more context = better
                    ),
                )
            else:
                # No critiques succeeded — pick iteration with most context tokens.
                with_result = [it for it in result.iterations if it.result is not None]
                if with_result:
                    best_iter = max(
                        with_result,
                        key=lambda it: int(it.context.token_count if it.context else 0),
                    )

        if best_iter is not None and best_iter.result is not None:
            result.final_output = best_iter.result.output
            if best_iter.faithfulness is not None and best_iter.faithfulness.should_abstain:
                result.final_output = build_abstention_output(task, best_iter.faithfulness)
            result.best_iteration = int(best_iter.iteration)
        result.total_latency_ms = _now_ms() - start_total

        final_quality = 0.0
        best_iter_num = 0
        if best_iter is not None:
            best_iter_num = best_iter.iteration
            if best_iter.critique:
                final_quality = float(best_iter.critique.quality_score or 0.0)

        final_tbl = Table(title="Run Summary")
        final_tbl.add_column("Iterations", justify="right")
        final_tbl.add_column("Best iter", justify="right")
        final_tbl.add_column("Converged", justify="center")
        final_tbl.add_column("Total ms", justify="right")
        final_tbl.add_column("Final quality", justify="right")
        final_tbl.add_row(
            str(len(result.iterations)),
            str(best_iter_num),
            "yes" if result.converged else "no",
            f"{result.total_latency_ms:.0f}",
            f"{final_quality:.2f}",
        )
        self.console.print(final_tbl)
        return result

    async def run_vanilla(self, task: str) -> PipelineResult:
        start_total = _now_ms()
        result = PipelineResult(task=task)

        self.console.rule("Vanilla Run")
        self.console.print(f"Task: {task}")

        executor = ModelExecutor(self.config.executor_model)
        iter_start = _now_ms()
        record = IterationRecord(iteration=1)
        result.iterations.append(record)

        try:
            exec_result = await executor.execute(task=task, context=None)
            record.result = exec_result
            record.context = None
            record.critique = None
            record.query_results = []
            record.specs = []
        except Exception as e:
            record.result = ExecutionResult(output=f"Error: vanilla execution failed: {e}")

        record.latency_ms = _now_ms() - iter_start
        result.final_output = record.result.output if record.result else ""
        result.total_latency_ms = _now_ms() - start_total
        result.converged = True
        result.best_iteration = 1
        return result


def pipeline_result_to_dict(res: PipelineResult) -> dict[str, Any]:
    """Best-effort conversion for logging/serialization."""

    try:
        return asdict(res)
    except Exception:
        # Avoid hard failures when some types have non-serializable fields.
        out: dict[str, Any] = {
            "task": res.task,
            "final_output": res.final_output,
            "total_latency_ms": res.total_latency_ms,
            "converged": res.converged,
            "best_iteration": int(res.best_iteration or 0),
            "iterations": [],
        }
        for it in res.iterations:
            out["iterations"].append(
                {
                    "iteration": it.iteration,
                    "latency_ms": it.latency_ms,
                    "specs": [asdict(s) for s in it.specs],
                    "query_results": [
                        {
                            "spec": asdict(q.spec),
                            "latency_ms": q.latency_ms,
                            "error": q.error,
                            "chunks": [
                                {
                                    "chunk_id": c.chunk_id,
                                    "score": c.score,
                                    "source": c.source,
                                    "token_count": c.token_count,
                                }
                                for c in (q.chunks or [])
                            ],
                        }
                        for q in (it.query_results or [])
                    ],
                    "context": None
                    if it.context is None
                    else {
                        "token_count": it.context.token_count,
                        "budget": it.context.budget,
                        "utilization": it.context.utilization,
                        "chunks_included": it.context.chunks_included,
                        "chunks_considered": it.context.chunks_considered,
                    },
                    "result": None
                    if it.result is None
                    else {
                        "model": it.result.model,
                        "latency_ms": it.result.latency_ms,
                        "tokens_prompt": it.result.tokens_prompt,
                        "tokens_completion": it.result.tokens_completion,
                    },
                    "critique": None
                    if it.critique is None
                    else {
                        "quality_score": it.critique.quality_score,
                        "context_utilization": it.critique.context_utilization,
                        "missing_info": list(it.critique.missing_info or []),
                        "irrelevant_chunks": list(it.critique.irrelevant_chunks or []),
                        "suggested_queries": list(it.critique.suggested_queries or []),
                    },
                    "faithfulness": None
                    if it.faithfulness is None
                    else {
                        "confidence": it.faithfulness.confidence,
                        "supported_ratio": it.faithfulness.supported_ratio,
                        "evidence_coverage_ratio": it.faithfulness.evidence_coverage_ratio,
                        "should_abstain": it.faithfulness.should_abstain,
                        "unsupported_claim_ids": list(it.faithfulness.unsupported_claim_ids),
                        "num_claims": len(it.faithfulness.claims or []),
                        "num_evidence": len(it.faithfulness.evidence or []),
                        "rationale": it.faithfulness.rationale,
                    },
                }
            )
        return out
