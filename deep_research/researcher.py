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


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


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
        *,
        run_dir: str | Path | None = None,
        resume: bool = False,
    ) -> str:
        """Conduct deep research on a query.

        If `run_dir` is provided, intermediate artifacts are checkpointed so that
        long runs can be resumed after failures.

        Args:
            query: The research query.
            on_status: Optional callback for status updates.
            run_dir: Directory to store intermediate artifacts.
            resume: If True, reuse existing artifacts from `run_dir` when present.

        Returns:
            Final research report as string.
        """

        def status(msg: str):
            if on_status:
                on_status(msg)

        run_path: Path | None = None
        if run_dir is not None:
            run_path = Path(run_dir)
            run_path.mkdir(parents=True, exist_ok=True)

        def artifact(name: str) -> Path | None:
            if run_path is None:
                return None
            return run_path / name

        brief_path = artifact('brief.txt')
        topics_path = artifact('topics.json')
        findings_path = artifact('findings.json')
        compressed_path = artifact('compressed.md')
        report_path = artifact('report.md')

        # Step 1: Check for clarification (optional)
        status('Analyzing query...')
        needs_clarification, message = await self.clarify(query)
        if needs_clarification:
            return f'Clarification needed: {message}'

        # Step 2: Create research brief
        if resume and brief_path and brief_path.exists():
            status('Loading research brief from checkpoint...')
            brief = _read_text(brief_path)
        else:
            status('Creating research brief...')
            brief = await self.create_brief(query)
            if brief_path:
                _write_text(brief_path, brief)
        status(f'Brief: {brief[:100]}...')

        # Step 3: Plan research
        if resume and topics_path and topics_path.exists():
            status('Loading research plan from checkpoint...')
            topics_data = _read_json(topics_path)
            topics = [ResearchTopic.model_validate(t) for t in topics_data]
        else:
            status('Planning research strategy...')
            topics = await self.plan_research(brief)
            if topics_path:
                _write_json(topics_path, [t.model_dump() for t in topics])
        status(f'Planned {len(topics)} research task(s)')

        # Step 4: Execute research in parallel
        if resume and findings_path and findings_path.exists():
            status('Loading research findings from checkpoint...')
            findings_data = _read_json(findings_path)
            findings = [ResearchFindings.model_validate(f) for f in findings_data]
        else:
            status('Conducting research...')
            research_tasks = [self.research_topic(topic) for topic in topics]
            findings = await asyncio.gather(*research_tasks)
            if findings_path:
                _write_json(findings_path, [f.model_dump() for f in findings])
        status(f'Completed {len(findings)} research task(s)')

        # Step 5: Compress findings
        if resume and compressed_path and compressed_path.exists():
            status('Loading compressed findings from checkpoint...')
            compressed = _read_text(compressed_path)
        else:
            status('Organizing findings...')
            compressed = await self.compress_findings(findings)
            if compressed_path:
                _write_text(compressed_path, compressed)

        # Step 6: Write final report
        if resume and report_path and report_path.exists():
            status('Loading final report from checkpoint...')
            report = _read_text(report_path)
        else:
            status('Writing final report...')
            report = await self.write_report(query, brief, compressed)
            if report_path:
                _write_text(report_path, report)

        status('Research complete!')
        return report
