# I-32: Workflow lacks gates and bounded success criteria per step

## Summary
The workflow proceeds through plan → parallel research → compress → report without validating intermediate outputs or bounding research effort in code, so empty/low-quality results and per-topic failures can silently propagate.

## Current behavior
- `max_search_iterations` exists but is unused.
- No gates validate brief/topic quality, source presence, or that compression/report include a Sources section.
- `asyncio.gather(*tasks)` fails the whole run if one topic errors.

## Recommended change
1. Add lightweight gates with actionable errors/fallbacks:
   - Brief: non-empty and above a minimum length.
   - Topics: at least 1 topic; each topic above a minimum length.
   - Research: enforce “has sources OR explicitly no sources found”.
   - Compress: require a Sources section (or move compressor to a structured output model).
2. Bound per-topic effort:
   - At minimum: pass `max_search_iterations` into the prompt and instruct the model to do ≤N searches.
   - Prefer: implement an explicit loop that allows at most 1 web search per iteration and accumulates notes/sources until `complete=True` or max iterations reached.
3. Improve parallel robustness:
   - Use `asyncio.gather(..., return_exceptions=True)` and continue with partial results.

## Acceptance criteria
- A single failed researcher task does not crash the entire workflow.
- Each step either passes a gate or returns a clear, user-visible failure message.
- Search effort is bounded by code (or at least by a prompt parameter that is actually used).
