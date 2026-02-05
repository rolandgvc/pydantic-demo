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
        workdir: str | Path | None = None,
        resume: bool = False,
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

        run_dir: Path | None = None
        if workdir is not None:
            run_dir = Path(workdir)
            run_dir.mkdir(parents=True, exist_ok=True)

        def checkpoint_path(name: str) -> Path | None:
            if run_dir is None:
                return None
            return run_dir / name

        def maybe_load_text(path: Path | None) -> str | None:
            if not resume or path is None or not path.exists():
                return None
            return path.read_text(encoding='utf-8')

        def save_text(path: Path | None, text: str) -> None:
            if path is None:
                return
            path.write_text(text, encoding='utf-8')

        # Step 1: Check for clarification (optional)
        status("Analyzing query...")
        needs_clarification, message = await self.clarify(query)
        if needs_clarification:
            return f"Clarification needed: {message}"

        # Step 2: Create research brief
        status("Creating research brief...")
        brief_path = checkpoint_path('brief.md')
        brief = maybe_load_text(brief_path)
        if brief is None:
            brief = await self.create_brief(query)
            save_text(brief_path, brief)
        status(f"Brief: {brief[:100]}...")

        # Step 3: Plan research
        status("Planning research strategy...")
        plan_path = checkpoint_path('plan.json')
        topics: list[ResearchTopic] | None = None
        if resume and plan_path is not None and plan_path.exists():
            plan_data = json.loads(plan_path.read_text(encoding='utf-8'))
            topics = [ResearchTopic.model_validate(t) for t in plan_data.get('topics', [])]
        if topics is None:
            topics = await self.plan_research(brief)
            if plan_path is not None:
                plan_path.write_text(
                    json.dumps({'topics': [t.model_dump() for t in topics]}, indent=2),
                    encoding='utf-8',
                )
        status(f"Planned {len(topics)} research task(s)")

        # Step 4: Execute research in parallel
        status("Conducting research...")
        findings: list[ResearchFindings] = []

        pending: list[tuple[int, ResearchTopic]] = []
        for i, topic in enumerate(topics):
            finding_path = checkpoint_path(f'finding_{i+1}.md')
            cached = maybe_load_text(finding_path)
            if cached is not None:
                findings.append(ResearchFindings(topic=topic.topic, findings=cached, sources=[]))
            else:
                pending.append((i, topic))

        if pending:
            tasks = [self.research_topic(topic) for _, topic in pending]
            results = await asyncio.gather(*tasks)
            for (i, topic), finding in zip(pending, results, strict=True):
                findings.append(finding)
                finding_path = checkpoint_path(f'finding_{i+1}.md')
                save_text(finding_path, finding.findings)

        status(f"Completed {len(findings)} research task(s)")

        # Step 5: Compress findings
        status("Organizing findings...")
        compressed_path = checkpoint_path('compressed.md')
        compressed = maybe_load_text(compressed_path)
        if compressed is None:
            compressed = await self.compress_findings(findings)
            save_text(compressed_path, compressed)

        # Step 6: Write final report
        status("Writing final report...")
        report_path = checkpoint_path('report.md')
        report = maybe_load_text(report_path)
        if report is None:
            report = await self.write_report(query, brief, compressed)
            save_text(report_path, report)

        status("Research complete!")
        return report
