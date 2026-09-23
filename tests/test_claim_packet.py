"""Development claim packets must stay tied to their source and assertion."""

import json
from pathlib import Path

import pytest

from scripts.validate_claim_packet import validate
from src.compiler.compiler import CNLCompiler


PACKET = Path(__file__).resolve().parents[1] / "data/benchmarks/fm2-0/development_partial_claims_20260922.json"
EXPANDED_PACKET = Path(__file__).resolve().parents[1] / "data/benchmarks/fm2-0/development_partial_claims_20260923.json"


def packet():
    return json.loads(PACKET.read_text(encoding="utf-8"))


def test_development_claim_is_source_traced_and_compiles():
    validate(packet())


def test_expanded_packet_has_reviewed_positives_and_abstentions():
    expanded = json.loads(EXPANDED_PACKET.read_text(encoding="utf-8"))
    validate(expanded)
    assert sum(row["decision"] == "positive" for row in expanded["claims"]) == 3
    assert sum(row["decision"] == "abstain" for row in expanded["claims"]) == 11


def test_source_span_cannot_be_invented():
    changed = packet()
    changed["claims"][0]["source_span"] = "Biometrics is a weapon."
    with pytest.raises(ValueError, match="source span absent"):
        validate(changed)


def test_gold_assertion_cannot_be_preloaded_as_a_declaration():
    changed = packet()
    changed["claims"][0]["declarations"] = ["(subclass BiometricsProcess Process)"]
    with pytest.raises(ValueError, match="class declaration"):
        validate(changed)


def test_compiled_kif_must_match():
    changed = packet()
    changed["claims"][0]["kif"] = "(subclass BiometricsProcess MilitaryProcess)"
    with pytest.raises(ValueError, match="compiled KIF mismatch"):
        validate(changed)


def test_new_class_is_not_global_background():
    with pytest.raises(ValueError, match="Unknown SUMO class"):
        CNLCompiler().compile("BiometricsProcess subclass-of Process", require_closed=True)


def test_abstention_cannot_hide_a_formula():
    expanded = json.loads(EXPANDED_PACKET.read_text(encoding="utf-8"))
    expanded["claims"][0]["cnl"] = "IntelligenceProcess subclass-of MilitaryProcess"
    with pytest.raises(ValueError, match="abstention cannot contain"):
        validate(expanded)


def test_abstention_requires_specific_reason():
    expanded = json.loads(EXPANDED_PACKET.read_text(encoding="utf-8"))
    expanded["claims"][0]["abstention_reason"] = "unclear"
    with pytest.raises(ValueError, match="abstention reason missing"):
        validate(expanded)
