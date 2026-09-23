from pathlib import Path

import pytest

from scripts.run_partial_claim_dev_probe import load_sources
from src.preprocessing.partial_claims import extract_definition_head


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter1-partial-claim-development-source"


def test_source_only_projection_and_exact_span_offsets():
    rows = load_sources(SOURCE / "sources.jsonl", SOURCE / "manifest.json")
    assert len(rows) == 14
    assert all("decision" not in row and "cnl" not in row for row in rows)
    by_paragraph = {row["paragraph"]: row for row in rows}
    row = by_paragraph["1-93"]
    result = extract_definition_head(row["source_excerpt"], row["source_sentence"])
    assert result.candidate_text == "Biometrics is the process"
    assert row["source_excerpt"][result.start:result.end] == result.source_span
    assert result.gate_reasons == ()


@pytest.mark.parametrize("sentence", [
    "Within Army doctrine, intelligence is a subset of information.",
    "A capability is a device or computer program.",
    "A report is not an order.",
    "Assess is part of an operation.",
    "Science is the application of scientific processes.",
    "The effectiveness of intelligence is measured by its outputs.",
])
def test_unsafe_or_non_kind_heads_abstain(sentence):
    result = extract_definition_head(sentence, sentence)
    assert result.status == "abstain"
    assert result.source_span is None


def test_source_prefix_must_be_exact():
    with pytest.raises(ValueError, match="exact paragraph prefix"):
        extract_definition_head("Biometrics is the process.", "Biometrics is a process.")
