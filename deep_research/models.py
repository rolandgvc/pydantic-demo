"""Data models for deep research."""
from pydantic import BaseModel, Field, HttpUrl


class ResearchTopic(BaseModel):
    """A research topic to investigate."""
    topic: str = Field(description="Detailed description of what to research (at least a paragraph)")


class ResearchFindings(BaseModel):
    """Findings from a research task."""

    topic: str
    findings: str = Field(description="Markdown findings for this topic")
    sources: list[HttpUrl] = Field(default_factory=list, description="Source URLs used")


class ResearchPlan(BaseModel):
    """Plan for conducting research."""

    topics: list[ResearchTopic] = Field(
        description="List of research topics to investigate in parallel",
        min_length=1,
        max_length=5,
    )
    reasoning: str = Field(description="Explanation of why these topics were chosen")


class ResearcherOutput(BaseModel):
    """Structured output produced by a researcher agent."""

    findings: str = Field(description="Markdown findings with inline citation markers like [1], [2]")
    sources: list[HttpUrl] = Field(
        default_factory=list,
        description="List of distinct source URLs referenced by the findings",
    )


class CompressedFindings(BaseModel):
    """Structured output produced by the compressor agent."""

    summary: str = Field(description="Cleaned, deduplicated markdown summary with inline citations")
    sources: list[HttpUrl] = Field(
        default_factory=list,
        description="List of distinct source URLs referenced by the summary",
    )


class ResearchComplete(BaseModel):
    """Signal that research is complete."""
    summary: str = Field(description="Brief summary of what was found")


class ClarificationNeeded(BaseModel):
    """Request for clarification from user."""
    need_clarification: bool
    question: str = ""
    acknowledgment: str = ""


class ResearchBrief(BaseModel):
    """Structured research brief from user query."""
    brief: str = Field(description="Detailed research question/brief")
    key_aspects: list[str] = Field(description="Key aspects to investigate")
