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

    def _checkpoint_path(self, checkpoint_dir: Path, name: str) -> Path:
        return checkpoint_dir / f"{name}.json"

    def _load_checkpoint(self, checkpoint_dir: Path, name: str) -> Any | None:
        path = self._checkpoint_path(checkpoint_dir, name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_checkpoint(self, checkpoint_dir: Path, name: str, data: Any) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(checkpoint_dir, name)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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

        checkpoint_path: Path | None = Path(checkpoint_dir) if checkpoint_dir else None
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_dir")

        if checkpoint_path is not None:
            # Save minimal run metadata for debugging.
            self._save_checkpoint(
                checkpoint_path,
                "run_meta",
                {"query": query, "date": get_today(), "model": self.model},
            )

        # Step 1: Check for clarification (optional)
        status("Analyzing query...")
        needs_clarification, message = await self.clarify(query)
        if needs_clarification:
            return f"Clarification needed: {message}"

        # Step 2: Create research brief
        status("Creating research brief...")
        brief_data = (
            self._load_checkpoint(checkpoint_path, "brief")
            if (resume and checkpoint_path is not None)
            else None
        )
        if brief_data is not None:
            brief = str(brief_data["brief"])
            status("Brief: (resumed from checkpoint)")
        else:
            brief = await self.create_brief(query)
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, "brief", {"brief": brief})
            status(f"Brief: {brief[:100]}...")

        # Step 3: Plan research
        status("Planning research strategy...")
        topics_data = (
            self._load_checkpoint(checkpoint_path, "topics")
            if (resume and checkpoint_path is not None)
            else None
        )
        if topics_data is not None:
            topics = [ResearchTopic.model_validate(t) for t in topics_data["topics"]]
            status(f"Planned {len(topics)} research task(s) (resumed)")
        else:
            topics = await self.plan_research(brief)
            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path,
                    "topics",
                    {"topics": [t.model_dump() for t in topics]},
                )
            status(f"Planned {len(topics)} research task(s)")

        # Step 4: Execute research in parallel
        status("Conducting research...")
        findings_data = (
            self._load_checkpoint(checkpoint_path, "findings")
            if (resume and checkpoint_path is not None)
            else None
        )
        if findings_data is not None:
            findings = [ResearchFindings.model_validate(f) for f in findings_data["findings"]]
            status(f"Completed {len(findings)} research task(s) (resumed)")
        else:
            research_tasks = [self.research_topic(topic) for topic in topics]
            findings = await asyncio.gather(*research_tasks)
            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path,
                    "findings",
                    {"findings": [f.model_dump() for f in findings]},
                )
            status(f"Completed {len(findings)} research task(s)")

        # Step 5: Compress findings
        status("Organizing findings...")
        compressed_data = (
            self._load_checkpoint(checkpoint_path, "compressed")
            if (resume and checkpoint_path is not None)
            else None
        )
        if compressed_data is not None:
            compressed = str(compressed_data["compressed"])
            status("Compressed findings: (resumed)")
        else:
            compressed = await self.compress_findings(findings)
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, "compressed", {"compressed": compressed})

        # Step 6: Write final report
        status("Writing final report...")
        report_data = (
            self._load_checkpoint(checkpoint_path, "report")
            if (resume and checkpoint_path is not None)
            else None
        )
        if report_data is not None:
            report = str(report_data["report"])
            status("Report: (resumed)")
        else:
            report = await self.write_report(query, brief, compressed)
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, "report", {"report": report})

        status("Research complete!")
        return report
