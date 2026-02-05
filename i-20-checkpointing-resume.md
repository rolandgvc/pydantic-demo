# I-20: Research workflow needs optional checkpointing and resume support

## Summary
Deep Research reruns all steps after any mid-run failure because it does not persist intermediate artifacts (brief, topics, per-topic findings, compressed synthesis). This increases cost and latency.

## Current code (where to look)
- `deep_research/researcher.py` runs the full pipeline in `DeepResearcher.research()` with no persistence.
- `deep_research/cli.py` exposes no flags for checkpointing/resume.

## Change plan
1. Add optional parameters to `DeepResearcher`:
   - `checkpoint_dir: Path | str | None = None`
   - `resume: bool = False`
2. Persist step outputs as JSON/text files after each completed stage:
   - `brief.txt`
   - `topics.json`
   - `findings.json`
   - `compressed.md`
   - `report.md`
3. When `resume=True`, if an artifact exists, load it and skip recomputation.
4. Add CLI flags:
   - `--checkpoint-dir <path>`
   - `--resume`

## Acceptance checks
- With checkpointing enabled, a failure after planning or research can be resumed without rerunning earlier steps.
- Default behavior unchanged when checkpointing disabled.
