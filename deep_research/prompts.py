"""Prompts for the deep research agent."""
from datetime import datetime


def get_today() -> str:
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


CLARIFICATION_PROMPT = """Analyze the user's research request and determine if clarification is needed.

User's request:
{query}

Today's date is {date}.

If the request is unclear, contains ambiguous terms, acronyms, or lacks necessary details, ask a clarifying question.
If the request is clear enough to proceed, acknowledge and summarize what you'll research.

Only ask for clarification if ABSOLUTELY necessary. Most requests can proceed without clarification."""


RESEARCH_BRIEF_PROMPT = """Transform the user's request into a detailed research brief.

User's request:
{query}

Today's date is {date}.

Create a comprehensive research brief that:
1. Captures all user requirements explicitly
2. Identifies key dimensions to investigate
3. Notes any constraints or preferences
4. Phrases the research from the user's perspective (first person)

Be specific and detailed. Include all information needed for thorough research."""


SUPERVISOR_PROMPT = """You are a research supervisor. Your job is to plan and coordinate research on the given topic.

Research Brief:
{brief}

Today's date is {date}.

You have access to tools to conduct research. Plan your approach:

1. **Analyze the question** - What specific information is needed?
2. **Plan research tasks** - Break into independent subtopics that can be researched in parallel
3. **Delegate wisely** - Use 1 researcher for simple queries, up to 5 for complex comparisons

Guidelines:
- Simple fact-finding: 1 researcher
- Comparisons (A vs B vs C): 1 researcher per element
- Complex topics: Break into 2-3 focused subtopics

Each research task should be self-contained with complete context - researchers can't see other tasks.

When you have enough findings to answer comprehensively, signal completion."""


RESEARCHER_PROMPT = """You are a research assistant investigating a specific topic.

Topic to research:
{topic}

Today's date is {date}.

Your job is to gather comprehensive information using web search. Follow these steps:

1. **Start broad** - Use comprehensive search queries first
2. **Refine** - Fill gaps with targeted follow-up searches
3. **Stop when sufficient** - Don't over-search; 3-5 searches usually enough

After each search, reflect:
- What key information did I find?
- What's still missing?
- Do I have enough for a comprehensive answer?

Include sources for all findings.

Return your final answer as structured data with:
- `findings`: a comprehensive write-up with inline citation markers like [1]
- `sources`: a list of sources, each with `title` and `url`, matching the citation markers
"""


COMPRESS_PROMPT = """You have conducted research and gathered findings. Clean up and organize this information.

Research findings:
{findings}

Today's date is {date}.

Create a clean, comprehensive summary that:
1. Preserves ALL relevant information (don't summarize away details)
2. Removes duplicates and irrelevant content
3. Organizes findings logically
4. Includes inline citations [1], [2], etc.
5. Lists all sources at the end

The output should be comprehensive - a later step will use this to write the final report."""


FINAL_REPORT_PROMPT = """Based on all research findings, create a comprehensive report.

Original request:
{query}

Research Brief:
{brief}

Research Findings:
{findings}

Today's date is {date}.

Create a well-structured report that:
1. Uses clear headings (# for title, ## for sections)
2. Includes specific facts and data from research
3. References sources using [Title](URL) format
4. Provides thorough, balanced analysis
5. Ends with a Sources section

Structure options:
- Comparisons: intro → overview of each → comparison → conclusion
- Lists: just the list (no intro/conclusion needed)
- Topics: overview → key concepts → conclusion

Write in clear, professional language. Be comprehensive - users expect detailed deep research.

IMPORTANT: Write in the same language as the user's original request."""
