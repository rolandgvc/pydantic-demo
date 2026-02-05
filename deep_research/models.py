"""Data models for deep research."""
from typing import Annotated
from pydantic import BaseModel, Field
from annotated_types import Ge, Le



class Source(BaseModel):
    """A source used to ground a research finding."""

    title: str = Field(description="Human-readable page/article title")
    url: str = Field(description="Canonical URL")


class ResearcherOutput(BaseModel):
    """Structured output expected from the researcher LLM call."""

    findings: str = Field(description="Research findings for the topic, with inline citation markers like [1]")
    sources: list[Source] = Field(default_factory=list, description="List of sources referenced in findings")

class ResearchTopic(BaseModel):
    """A research topic to investigate."""
    topic: str = Field(description="Detailed description of what to research (at least a paragraph)")


class ResearchFindings(BaseModel):
    """Findings from a research task."""

    topic: str
    findings: str
    sources: list[Source] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Plan for conducting research."""
    topics: list[ResearchTopic] = Field(
        description="List of research topics to investigate in parallel",
        min_length=1,
        max_length=5
    )
    reasoning: str = Field(description="Explanation of why these topics were chosen")


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
