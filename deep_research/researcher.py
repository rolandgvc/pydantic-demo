"""Deep research implementation using Pydantic AI."""
import asyncio
import re
from dataclasses import dataclass
from typing import Any

import logfire
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from .models import (
    ClarificationNeeded,
    ResearchBrief,
    ResearchFindings,
    ResearchPlan,
    ResearchTopic,
)
from .prompts import (
    CLARIFICATION_PROMPT,
    COMPRESS_PROMPT,
    FINAL_REPORT_PROMPT,
    RESEARCHER_PROMPT,
    RESEARCH_BRIEF_PROMPT,
    SUPERVISOR_PROMPT,
    get_today,
)


@dataclass
class ResearchContext:
    """Context for research operations."""
    query: str
    brief: str = ""
    findings: list[ResearchFindings] = None

    def __post_init__(self):
        if self.findings is None:
            self.findings = []




class StepValidationError(RuntimeError):
    """Raised when a workflow step violates an expected contract."""

    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message


_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")


def extract_urls(text: str) -> list[str]:
    """Best-effort URL extraction for simple source gating.

    This is intentionally lightweight: the system currently asks for inline citations,
    but later steps need *some* deterministic signal that sources exist.
    """

    urls = _URL_RE.findall(text or "")
    # Normalize and de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

class DeepResearcher:
    """Deep research agent using Pydantic AI."""

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        max_parallel_researchers: int = 3,
        max_search_iterations: int = 5,
        allow_clarification: bool = False,
    ):
        self.model = model
        self.max_parallel_researchers = max_parallel_researchers
        self.max_search_iterations = max_search_iterations
        self.allow_clarification = allow_clarification

        # Initialize agents
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents."""
        # Clarification agent
        self.clarifier = Agent(
            self.model,
            output_type=ClarificationNeeded,
            system_prompt="You help clarify research requests when needed.",
        )

        # Brief writer agent
        self.brief_writer = Agent(
            self.model,
            output_type=ResearchBrief,
            system_prompt="You transform user queries into detailed research briefs.",
        )

        # Supervisor agent - plans research
        self.supervisor = Agent(
            self.model,
            output_type=ResearchPlan,
            system_prompt="You are a research supervisor who plans research strategies.",
        )

        # Researcher agent - does actual searching
        self.researcher = Agent(
            self.model,
            tools=[duckduckgo_search_tool()],
            system_prompt="You are a research assistant who searches for information.",
        )

        # Compressor agent - summarizes findings
        self.compressor = Agent(
            self.model,
            system_prompt="You compress and organize research findings.",
        )

        # Report writer agent
        self.report_writer = Agent(
            self.model,
            system_prompt="You write comprehensive research reports.",
        )

    @logfire.instrument('Clarify query')
    async def clarify(self, query: str) -> tuple[bool, str]:
        """Check if clarification is needed.

        Returns:
            Tuple of (needs_clarification, message)
        """
        if not self.allow_clarification:
            return False, ""

        prompt = CLARIFICATION_PROMPT.format(query=query, date=get_today())
        result = await self.clarifier.run(prompt)

        if result.output.need_clarification:
            return True, result.output.question
        return False, result.output.acknowledgment

    @logfire.instrument('Create research brief')
    async def create_brief(self, query: str) -> str:
        """Transform query into research brief."""
        prompt = RESEARCH_BRIEF_PROMPT.format(query=query, date=get_today())
        result = await self.brief_writer.run(prompt)
        return result.output.brief

    @logfire.instrument('Plan research')
    async def plan_research(self, brief: str) -> list[ResearchTopic]:
        """Plan research by breaking into subtopics."""
        prompt = SUPERVISOR_PROMPT.format(brief=brief, date=get_today())
        result = await self.supervisor.run(prompt)

        # Limit to max parallel researchers
        topics = result.output.topics[:self.max_parallel_researchers]
        return topics

    @logfire.instrument('Research topic: {topic.topic}')
    async def research_topic(self, topic: ResearchTopic) -> ResearchFindings:
        """Research a single topic."""
        prompt = RESEARCHER_PROMPT.format(topic=topic.topic, date=get_today())
        result = await self.researcher.run(prompt)

        # Extract findings from conversation
        findings = result.output if isinstance(result.output, str) else str(result.output)
        sources = extract_urls(findings)

        if not findings.strip():
            raise StepValidationError("research_topic", f"Empty findings for topic: {topic.topic!r}")
        if not sources:
            raise StepValidationError(
                "research_topic",
                f"No source URLs detected in findings for topic: {topic.topic!r}",
            )

        return ResearchFindings(
            topic=topic.topic,
            findings=findings,
            sources=sources,
        )

    @logfire.instrument('Compress findings')
    async def compress_findings(self, findings: list[ResearchFindings]) -> str:
        """Compress multiple findings into organized summary."""
        all_findings = "\n\n---\n\n".join([
            f"## {f.topic}\n\n{f.findings}"
            for f in findings
        ])

        prompt = COMPRESS_PROMPT.format(findings=all_findings, date=get_today())
        result = await self.compressor.run(prompt)
        return result.output if isinstance(result.output, str) else str(result.output)

    @logfire.instrument('Write final report')
    async def write_report(self, query: str, brief: str, findings: str) -> str:
        """Write final comprehensive report."""
        prompt = FINAL_REPORT_PROMPT.format(
            query=query,
            brief=brief,

            findings=findings,
            date=get_today()
        )
        result = await self.report_writer.run(prompt)
        return result.output if isinstance(result.output, str) else str(result.output)

    @logfire.instrument('Deep research: {query}')
    async def research(
        self,
        query: str,
        on_status: callable = None,
    ) -> str:
        """Conduct deep research on a query.

        Args:
            query: The research query
            on_status: Optional callback for status updates

        Returns:
            Final research report as string
        """
        def status(msg: str):
            if on_status:
                on_status(msg)

        # Step 1: Check for clarification (optional)
        status("Analyzing query...")
        needs_clarification, message = await self.clarify(query)
        if needs_clarification:
            return f"Clarification needed: {message}"

        # Step 2: Create research brief
        status("Creating research brief...")
        brief = await self.create_brief(query)
        status(f"Brief: {brief[:100]}...")

        # Step 3: Plan research
        status("Planning research strategy...")
        topics = await self.plan_research(brief)
        status(f"Planned {len(topics)} research task(s)")

        # Step 4: Execute research in parallel

        if not topics:
            raise StepValidationError("plan_research", "Supervisor returned no topics")
        if any(not t.topic.strip() for t in topics):
            raise StepValidationError("plan_research", "One or more planned topics were empty")

        status("Conducting research...")
        research_tasks = [self.research_topic(topic) for topic in topics]
        findings = await asyncio.gather(*research_tasks)
        status(f"Completed {len(findings)} research task(s)")



        # Step 5: Compress findings
        status("Organizing findings...")
        compressed = await self.compress_findings(findings)

        # Step 6: Write final report

        if not extract_urls(compressed):
            raise StepValidationError(
                "compress_findings",
                "No source URLs detected in compressed findings; cannot produce a grounded report",
            )

        status("Writing final report...")
        report = await self.write_report(query, brief, compressed)


        if not report.strip():
            raise StepValidationError("write_report", "Final report was empty")
        if not extract_urls(report):
            raise StepValidationError(
                "write_report",
                "No source URLs detected in final report; expected a Sources section with links",
            )

        status("Research complete!")






        return report
