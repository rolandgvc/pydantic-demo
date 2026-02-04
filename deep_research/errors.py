"""Stage-specific errors to provide actionable attribution."""

from __future__ import annotations


class DeepResearchError(Exception):
    """Base class for deep-research errors."""


class ClarificationError(DeepResearchError):
    pass


class BriefGenerationError(DeepResearchError):
    pass


class PlanningError(DeepResearchError):
    pass


class ResearchTopicError(DeepResearchError):
    pass


class CompressionError(DeepResearchError):
    pass


class ReportWritingError(DeepResearchError):
    pass
