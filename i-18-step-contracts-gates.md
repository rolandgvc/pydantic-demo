# I-18: Workflow lacks step contracts and programmatic gates

## Summary
Several Deep Research stages return untyped strings and the orchestrator does not validate intermediate outputs. This allows empty/low-quality results (e.g., missing sources) to propagate into the final report.

## Current code (where to look)
- `deep_research/researcher.py`:
  - `research_topic()` always returns `sources=[]`
  - `compress_findings()` and `write_report()` return strings
  - `research()` has no validation gates between stages
- `deep_research/models.py` defines some schemas but not for later steps.

## Change plan
1. Add a small set of schemas to `models.py` for later steps:
   - `Source` (title, url)
   - `TopicFindings` (topic, summary/findings, sources)
   - `CompressedFindings` (summary text + sources)
2. Update agents in `_init_agents()` to use `output_type` for researcher + compressor where feasible.
3. Add lightweight programmatic gates in `DeepResearcher.research()`:
   - planned topics non-empty
   - each topic has at least 1 source
   - compressed findings has non-empty sources
4. Fail fast with step-specific exceptions/messages so the user sees what broke.

## Acceptance checks
- `sources` are populated from researcher output (structured).
- Pipeline raises a clear error when a gate fails (instead of producing an empty report).
- CLI behavior remains backwards compatible (still prints a report or a clear error).
