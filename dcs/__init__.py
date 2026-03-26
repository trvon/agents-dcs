"""Dynamic Context Scaffold — YAMS-backed retrieval harness for small LLM agents."""

__version__ = "0.0.1"

from .plan_review import PlanReviewer

__all__ = ["PlanReviewer", "__version__"]
