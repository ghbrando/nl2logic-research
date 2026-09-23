"""Development claim packets must stay tied to their source and assertion."""

import json
from pathlib import Path

import pytest

from scripts.validate_claim_packet import _text_sha256, validate
from src.compiler.compiler import CNLCompiler


PACKET = Path(__file__).resolve().parents[1] / "data/benchmarks/fm2-0/development_partial_claims_20260922.json"
EXPANDED_PACKET = Path(__file__).resolve().parents[1] / "data/benchmarks/fm2-0/development_partial_claims_20260923.json"
EVAL_PACKET = Path(__file__).resolve().parents[1] / "data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/ai_reviewed_claims_20260923.json"


def packet():
    return json.loads(PACKET.read_text(encoding="utf-8"))


def test_development_claim_is_source_traced_and_compiles():
    validate(packet())


def test_expanded_packet_has_reviewed_positives_and_abstentions():
    expanded = json.loads(EXPANDED_PACKET.read_text(encoding="utf-8"))
    validate(expanded)
    assert sum(row["decision"] == "positive" for row in expanded["claims"]) == 3
    assert sum(row["decision"] == "abstain" for row in expanded["claims"]) == 11


def test_frozen_eval_labels_cover_every_selected_passage():
    reviewed = json.loads(EVAL_PACKET.read_text(encoding="utf-8"))
    validate(reviewed)
    assert sum(row["decision"] == "positive" for row in reviewed["claims"]) == 1
    assert sum(row["decision"] == "abstain" for row in reviewed["claims"]) == 28


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


def test_eval_label_packet_rejects_changed_source_manifest():
    reviewed = json.loads(EVAL_PACKET.read_text(encoding="utf-8"))
    reviewed["source"]["selection_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Selection manifest hash mismatch"):
        validate(reviewed)


def test_eval_label_packet_rejects_missing_passage():
    reviewed = json.loads(EVAL_PACKET.read_text(encoding="utf-8"))
    reviewed["claims"].pop()
    with pytest.raises(ValueError, match="cover every selected source passage"):
        validate(reviewed)


def test_eval_label_packet_requires_frozen_status():
    reviewed = json.loads(EVAL_PACKET.read_text(encoding="utf-8"))
    reviewed["label_status"] = "draft"
    with pytest.raises(ValueError, match="must be frozen"):
        validate(reviewed)


def test_frozen_text_hash_is_line_ending_neutral(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"value": 1}\r\n')
    windows_hash = _text_sha256(artifact)
    artifact.write_bytes(b'{"value": 1}\n')
    assert _text_sha256(artifact) == windows_hash
