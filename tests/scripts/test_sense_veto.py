from pathlib import Path

from scripts.build_sense_veto_pairs import HELDOUT_CLASS_SENSES, HELDOUT_WORDS, TRAIN_SENSES, build
from scripts.run_sense_veto import dev_claims
from src.ingest.sense_veto import accept_target, parent_gloss, reject_target, veto_prompt

ROOT = Path(__file__).resolve().parents[2]


def test_veto_prompt_carries_sentence_claim_and_gloss():
    prompt = veto_prompt("Relay audit is a product of analysis.", "RelayAuditProduct", "Product")
    assert "Source: Relay audit is a product of analysis." in prompt
    assert "Claim: RelayAuditProduct subclass-of Product" in prompt
    assert "Parent meaning: An Artifact that is produced by Manufacture." in prompt
    assert reject_target("A", "B") == "not " + accept_target("A", "B")
    assert ";;" not in parent_gloss("Document")


def test_false_friend_classes_are_held_out_of_training_senses():
    assert {parent for parent, _, _ in TRAIN_SENSES}.isdisjoint({"Product", "Cycle", "Sensor"})
    rows = build(HELDOUT_CLASS_SENSES, HELDOUT_WORDS, seed=13, per_template=2)
    assert {row["parent"] for row in rows} == {"Product", "Cycle"}
    assert {row["label"] for row in rows} == {"accept", "reject"}


def test_development_claims_are_gate_accepted_rows():
    runs = ROOT / "results/diagnostics"
    claims = dev_claims([runs / "appendix_definition_cpu_20260923/scoring/scored.jsonl"])
    assert {(c["paragraph"], c["parent"], c["label_match"]) for c in claims} == {
        ("B-18", "Product", False), ("B-39", "Process", True), ("B-44", "Process", True)}
    assert all(c["run"] == "appendix_definition_cpu_20260923" for c in claims)
