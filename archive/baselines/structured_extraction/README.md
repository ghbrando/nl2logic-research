# Structured Extraction Baseline

This directory contains the archived LLM structured-extraction baseline that previously lived in:

- `src/ingest/structured_extractor.py`
- `scripts/test_structured_extraction.py`

It is preserved for comparison and historical reference, but it is not part of the main doctrine ingest path anymore.

Why it was archived:

- the live pipeline does not import or call it
- the main research contribution is the constrained T5/FSM path
- keeping this code under `src/ingest` made the repo look like it had two equal main pipelines when it does not

Use it only as:

- a comparison baseline
- a reference for earlier experiments
- a possible source of silver-label ideas
