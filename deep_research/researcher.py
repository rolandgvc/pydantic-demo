"""Deep research implementation using Pydantic AI."""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
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


class DeepResearcher:
    """Deep research agent using Pydantic AI."""

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        max_parallel_researchers: int = 3,
        max_search_iterations: int = 5,
        allow_clarification: bool = False,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ):
        self.model = model
        self.max_parallel_researchers = max_parallel_researchers

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.resume = resume
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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


    def _ckpt_path(self, name: str) -> Path | None:
        if not self.checkpoint_dir:
            return None
        return self.checkpoint_dir / name

    def _save_text(self, name: str, text: str) -> None:
        path = self._ckpt_path(name)
        if not path:
            return
        path.write_text(text, encoding="utf-8")

    def _load_text(self, name: str) -> str | None:
        path = self._ckpt_path(name)
        if not path or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _save_json(self, name: str, obj: Any) -> None:
        path = self._ckpt_path(name)
        if not path:
            return
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_json(self, name: str) -> Any | None:
        path = self._ckpt_path(name)
        if not path or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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

        return ResearchFindings(
            topic=topic.topic,
            findings=findings,
            sources=[]  # Sources are inline in the findings
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

        if self.checkpoint_dir:
            status(
                f"Checkpointing: dir={self.checkpoint_dir} resume={self.resume}"  # pragma: no cover
            )

        # Step 2: Create research brief
        status("Creating research brief...")
        brief = None
        if self.resume:
            brief = self._load_text("brief.txt")
            if brief is not None:
                status("Loaded brief from checkpoint")
        if brief is None:
            brief = await self.create_brief(query)
            self._save_text("brief.txt", brief)
        status(f"Brief: {brief[:100]}...")

        # Step 3: Plan research
        status("Planning research strategy...")
        topics: list[ResearchTopic] | None = None
        if self.resume:
            topics_data = self._load_json("topics.json")
            if topics_data is not None:
                topics = [ResearchTopic.model_validate(t) for t in topics_data]
                status("Loaded topics from checkpoint")
        if topics is None:
            topics = await self.plan_research(brief)
            self._save_json("topics.json", [t.model_dump() for t in topics])
        status(f"Planned {len(topics)} research task(s)")

        # Step 4: Execute research in parallel
        status("Conducting research...")
        findings: list[ResearchFindings] | None = None
        if self.resume:
            findings_data = self._load_json("findings.json")
            if findings_data is not None:
                findings = [ResearchFindings.model_validate(f) for f in findings_data]
                status("Loaded findings from checkpoint")
        if findings is None:
            research_tasks = [self.research_topic(topic) for topic in topics]
            findings = await asyncio.gather(*research_tasks)
            self._save_json("findings.json", [f.model_dump() for f in findings])
        status(f"Completed {len(findings)} research task(s)")

        # Step 5: Compress findings
        status("Organizing findings...")
        compressed = None
        if self.resume:
            compressed = self._load_text("compressed.md")
            if compressed is not None:
                status("Loaded compressed findings from checkpoint")
        if compressed is None:
            compressed = await self.compress_findings(findings)
            self._save_text("compressed.md", compressed)

        # Step 6: Write final report
        status("Writing final report...")
        report = None
        if self.resume:
            report = self._load_text("report.md")
            if report is not None:
                status("Loaded final report from checkpoint")
        if report is None:
            report = await self.write_report(query, brief, compressed)
            self._save_text("report.md", report)

        status("Research complete!")
        return report
