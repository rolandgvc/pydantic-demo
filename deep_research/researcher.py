"""Deep research implementation using Pydantic AI."""
import asyncio
from dataclasses import dataclass
import hashlib
import json
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
        checkpoint_dir: str | Path | None = None,
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
        if checkpoint_dir is not None:
            run_hash = hashlib.sha256(f"{self.model}\n{query}".encode('utf-8')).hexdigest()[:12]
            run_dir = Path(checkpoint_dir) / f"run-{run_hash}"
            run_dir.mkdir(parents=True, exist_ok=True)

        def _json_path(name: str) -> Path:
            assert run_dir is not None
            return run_dir / f"{name}.json"

        def _text_path(name: str) -> Path:
            assert run_dir is not None
            return run_dir / f"{name}.md"

        def _save_json(name: str, data: Any) -> None:
            if run_dir is None:
                return
            _json_path(name).write_text(json.dumps(data, indent=2, ensure_ascii=False))

        def _load_json(name: str) -> Any | None:
            if run_dir is None:
                return None
            p = _json_path(name)
            if resume and p.exists():
                return json.loads(p.read_text())
            return None

        def _save_text(name: str, text: str) -> None:
            if run_dir is None:
                return
            _text_path(name).write_text(text)

        def _load_text(name: str) -> str | None:
            if run_dir is None:
                return None
            p = _text_path(name)
            if resume and p.exists():
                return p.read_text()
            return None

        # Step 1: Check for clarification (optional)
        status("Analyzing query...")
        needs_clarification, message = await self.clarify(query)
        if needs_clarification:
            return f"Clarification needed: {message}"

        # Step 2: Create research brief
        status("Creating research brief...")
        brief_data = _load_json('brief')
        if brief_data is not None:
            brief = str(brief_data.get('brief', ''))
            status("Loaded brief from checkpoint")
        else:
            brief = await self.create_brief(query)
            _save_json('brief', {'brief': brief})
        status(f"Brief: {brief[:100]}...")

        # Step 3: Plan research
        status("Planning research strategy...")
        topics_data = _load_json('topics')
        if topics_data is not None:
            topics = [ResearchTopic.model_validate(t) for t in topics_data]
            status("Loaded topics from checkpoint")
        else:
            topics = await self.plan_research(brief)
            _save_json('topics', [t.model_dump() for t in topics])
        status(f"Planned {len(topics)} research task(s)")

        # Step 4: Execute research in parallel
        status("Conducting research...")
        findings_data = _load_json('findings')
        if findings_data is not None:
            findings = [ResearchFindings.model_validate(f) for f in findings_data]
            status("Loaded findings from checkpoint")
        else:
            research_tasks = [self.research_topic(topic) for topic in topics]
            findings = await asyncio.gather(*research_tasks)
            _save_json('findings', [f.model_dump() for f in findings])
        status(f"Completed {len(findings)} research task(s)")

        # Step 5: Compress findings
        status("Organizing findings...")
        compressed = _load_text('compressed')
        if compressed is not None:
            status("Loaded compressed findings from checkpoint")
        else:
            compressed = await self.compress_findings(findings)
            _save_text('compressed', compressed)

        # Step 6: Write final report
        status("Writing final report...")
        report = _load_text('report')
        if report is not None:
            status("Loaded final report from checkpoint")
        else:
            report = await self.write_report(query, brief, compressed)
            _save_text('report', report)

        status("Research complete!")
        return report
