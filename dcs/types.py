"""Shared types for the Dynamic Context Scaffold pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Query / Decomposition types
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Kind of YAMS query to execute."""

    SEMANTIC = "semantic"
    GREP = "grep"
    GRAPH = "graph"
    GET = "get"
    LIST = "list"


@dataclass
class QuerySpec:
    """A single information need produced by the decomposer."""

    query: str
    query_type: QueryType
    importance: float  # 0.0–1.0
    reason: str = ""


# ---------------------------------------------------------------------------
# YAMS result types
# ---------------------------------------------------------------------------


@dataclass
class YAMSChunk:
    """A single chunk/result returned by YAMS."""

    chunk_id: str  # hash or path
    content: str
    score: float = 0.0
    source: str = ""  # origin path or collection
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0  # filled by assembler


@dataclass
class YAMSQueryResult:
    """Aggregated result of a single YAMS query."""

    spec: QuerySpec
    chunks: list[YAMSChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Context assembly types
# ---------------------------------------------------------------------------


@dataclass
class ContextBlock:
    """Token-bounded assembled context ready for model injection."""

    content: str
    sources: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    token_count: int = 0
    budget: int = 0
    utilization: float = 0.0  # token_count / budget
    chunks_included: int = 0
    chunks_considered: int = 0


# ---------------------------------------------------------------------------
# Execution types
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result from running a model on assembled context."""

    output: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    model: str = ""
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Critique / Feedback types
# ---------------------------------------------------------------------------


@dataclass
class Critique:
    """Self-critique feedback on context quality and output."""

    context_utilization: float  # 0.0–1.0
    missing_info: list[str] = field(default_factory=list)
    irrelevant_chunks: list[str] = field(default_factory=list)  # chunk_ids
    quality_score: float = 0.0  # 0.0–1.0
    suggested_queries: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# No-ground-truth faithfulness types
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """Deterministic evidence extracted from assembled context."""

    evidence_id: str
    source: str
    snippet: str
    chunk_id: str = ""


@dataclass
class ClaimItem:
    """A normalized claim extracted from model output."""

    claim_id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    supported: bool = False
    confidence: float = 0.0


@dataclass
class FaithfulnessReport:
    """Ground-truth-free faithfulness checks for a generated answer."""

    claims: list[ClaimItem] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    unsupported_claim_ids: list[str] = field(default_factory=list)
    supported_ratio: float = 0.0
    evidence_coverage_ratio: float = 0.0
    confidence: float = 0.0
    should_abstain: bool = False
    rationale: str = ""


# ---------------------------------------------------------------------------
# Plan review types
# ---------------------------------------------------------------------------


class PlanStepStatus(str, Enum):
    """Review status for a single plan step."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    UNVERIFIED = "unverified"


@dataclass
class PlanStep:
    """A normalized implementation-plan step."""

    step_id: str
    description: str
    section: str = ""
    step_type: str = "step"
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class PlanStepReview:
    """Review verdict for a single plan step."""

    step_id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.UNVERIFIED
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class PlanReviewInput:
    """Inputs for post-execution plan-vs-change review."""

    plan: str
    task: str = ""
    diff_text: str = ""
    change_summary: str = ""
    execution_summary: str = ""
    changed_files: list[str] = field(default_factory=list)


@dataclass
class PlanReviewResult:
    """Structured review of whether code changes satisfied a plan."""

    task: str
    plan_steps: list[PlanStep] = field(default_factory=list)
    step_reviews: list[PlanStepReview] = field(default_factory=list)
    coverage_score: float = 0.0
    executed_well: bool = False
    summary: str = ""
    gaps: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    suggested_tests: list[str] = field(default_factory=list)
    followup_plan: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    ingested_artifacts: list[str] = field(default_factory=list)
    context: ContextBlock | None = None
    query_results: list[YAMSQueryResult] = field(default_factory=list)
    model_output: str = ""
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Pipeline types
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Stages of the DCS pipeline."""

    DECOMPOSE = "decompose"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    ASSEMBLE = "assemble"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    OPTIMIZE = "optimize"


@dataclass
class IterationRecord:
    """Record of a single pipeline iteration."""

    iteration: int
    specs: list[QuerySpec] = field(default_factory=list)
    query_results: list[YAMSQueryResult] = field(default_factory=list)
    context: ContextBlock | None = None
    result: ExecutionResult | None = None
    critique: Critique | None = None
    faithfulness: FaithfulnessReport | None = None
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    """Full result of a DCS pipeline run (possibly multi-iteration)."""

    task: str
    iterations: list[IterationRecord] = field(default_factory=list)
    final_output: str = ""
    total_latency_ms: float = 0.0
    converged: bool = False
    best_iteration: int = 0
    plan_review: PlanReviewResult | None = None

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def final_critique(self) -> Critique | None:
        if self.iterations:
            return self.iterations[-1].critique
        return None


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a model backend."""

    name: str  # model identifier (e.g. "qwen/qwen3-4b-thinking-2507")
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"  # LM Studio default
    context_window: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.7
    # Optional suffix appended to system prompts. Use "/no_think" for qwen3
    # thinking models to disable chain-of-thought and save output tokens.
    system_suffix: str = ""
    # Execution controls
    request_timeout_s: float = 600.0
    max_retries: int = 2
    retry_backoff_s: float = 2.0


@dataclass
class PipelineConfig:
    """Configuration for a DCS pipeline run."""

    # Model settings
    executor_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="qwen/qwen3-4b-thinking-2507")
    )
    critic_model: ModelConfig | None = None  # defaults to executor_model

    # Context budget
    context_budget: int = 2048  # tokens reserved for retrieved context
    system_prompt_budget: int = 512  # tokens for system prompt
    output_reserve: int = 1024  # tokens reserved for model output
    codemap_budget: int = 256  # tokens reserved for structural codemap prefix
    codemap_max_files: int = 5
    codemap_max_symbols_per_file: int = 8
    codemap_include_type_counts: bool = False
    min_context_budget: int = 128  # lower bound for overflow retries
    min_output_tokens: int = 256  # lower bound for overflow retries
    context_shrink_factor: float = 0.7  # shrink multiplier on overflow retry
    max_context_overflow_retries: int = 2

    # Context profile behavior
    # - auto: apply large profile when detected context window >= large_context_threshold
    # - standard: keep configured defaults
    # - large: force large profile
    context_profile: str = "auto"
    large_context_threshold: int = 12288
    large_context_budget: int = 4096
    large_system_prompt_budget: int = 768
    large_output_reserve: int = 1536
    large_codemap_budget: int = 512

    # Retrieval settings
    max_queries_per_iteration: int = 5
    max_chunks_per_query: int = 10
    min_chunk_score: float = 0.1
    retrieval_max_concurrency: int = 2
    use_dspy_retrieval_rerank: bool = False
    dspy_retrieval_top_k: int = 5
    dspy_retrieval_max_tokens: int = 16384
    dspy_retrieval_demo_count: int = 0
    dspy_retrieval_prefer_json: bool = True
    dspy_retrieval_optimize: bool = False
    dspy_retrieval_optimizer_trainset_size: int = 6
    dspy_retrieval_bootstrapped_demos: int = 2
    dspy_retrieval_labeled_demos: int = 4
    dspy_retrieval_metric_threshold: float = 0.75
    dspy_retrieval_model: ModelConfig | None = None

    # Iteration settings
    max_iterations: int = 3
    quality_threshold: float = 0.7  # stop if critique quality >= this
    convergence_delta: float = 0.05  # stop if quality improvement < this

    # Ground-truth-free faithfulness mode
    no_ground_truth_mode: bool = True
    use_dspy_faithfulness: bool = True
    claim_evidence_min_overlap: float = 0.12
    faithfulness_min_confidence: float = 0.60
    faithfulness_max_unsupported_ratio: float = 0.40
    faithfulness_min_supported_claims: int = 1

    # Retrieval planning behavior
    # When disabled, skip task-specific hardcoded query seeding so benchmarks
    # can measure general retrieval behavior without task-family leakage.
    enable_task_seeding: bool = True

    # YAMS settings
    yams_binary: str = "yams"
    yams_data_dir: str | None = None
    yams_cwd: str | None = None  # scope search/grep to this directory tree

    # Plan-review settings
    plan_review_context_budget: int = 1536
    plan_review_max_changed_files: int = 8
    plan_review_search_limit: int = 4

    # Search weight overrides (passed to YAMS search)
    search_weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Category of evaluation task."""

    CODING = "coding"
    QA = "qa"
    AGENT = "agent"


class EvalMetric(str, Enum):
    """Evaluation metric names."""

    TASK_SUCCESS = "task_success"
    CONTEXT_EFFICIENCY = "context_efficiency"
    RETRIEVAL_PRECISION = "retrieval_precision"
    ITERATIONS_TO_CONVERGE = "iterations_to_converge"
    TOTAL_LATENCY = "total_latency"
    TOKEN_COST = "token_cost"
    PLAN_COVERAGE = "plan_coverage"
    PLAN_EXECUTED_WELL = "plan_executed_well"


@dataclass
class EvalTask:
    """Definition of a single evaluation task."""

    id: str
    task_type: TaskType
    description: str
    plan: str = ""
    ground_truth: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task_id: str
    pipeline_result: PipelineResult | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None
    task_type: str = ""
    tags: list[str] = field(default_factory=list)
    repeat_index: int = 1


@dataclass
class ComparisonResult:
    """Comparison of scaffolded vs baseline across a task suite."""

    config_name: str
    model: str
    scaffolded: bool
    tasks: list[EvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.passed) / len(self.tasks)

    @property
    def avg_metric(self) -> dict[str, float]:
        if not self.tasks:
            return {}
        metrics: dict[str, list[float]] = {}
        for t in self.tasks:
            for k, v in t.metrics.items():
                metrics.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in metrics.items()}
