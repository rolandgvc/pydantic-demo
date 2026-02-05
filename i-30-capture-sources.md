# I-30: Research findings do not capture or validate sources

## Summary
The workflow asks researchers to include sources/citations, but the implementation discards them (`sources=[]`), so later steps can’t reliably ground claims or produce consistent references.

## Current behavior
- `DeepResearcher.research_topic()` converts the researcher agent output to a string and always returns `ResearchFindings(..., sources=[])`.
- The researcher agent has no `output_type`, so there is no structured contract for sources.

Relevant code:
- `deep_research/researcher.py` (research_topic)

## Recommended change
1. Update `deep_research/models.py`:
   - Add a `Source` model with `title: str` and `url: AnyUrl` (or `str` with validation).
   - Update `ResearchFindings` to include `sources: list[Source]` and (optionally) a more structured `findings` field.
2. Update `deep_research/researcher.py`:
   - Configure `self.researcher = Agent(..., output_type=ResearchFindings, tools=[duckduckgo_search_tool()])`.
   - Update `RESEARCHER_PROMPT` to instruct the model to return findings + structured sources.
   - Add a small gate after `result.output` to ensure at least one source exists; otherwise add a sentinel source or a clear “no sources found” marker.
3. (Optional, but recommended) Update `COMPRESS_PROMPT` to preserve and normalize citations/sources using the structured `sources` list.

## Acceptance criteria
- `ResearchFindings.sources` is populated with at least one URL for typical queries.
- Compressor/final report can rely on the structured sources list rather than ad-hoc inline links.
