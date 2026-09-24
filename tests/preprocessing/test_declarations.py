import json
from pathlib import Path

import pytest

from src.preprocessing.declarations import load_declaration_registry, match_declaration


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter1-partial-claim-development-source"


def _registry():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    return load_declaration_registry(
        SOURCE / "provisional_declarations.json",
        source_packet_sha256=manifest["packet_sha256"],
    )


def test_registry_contains_only_isolated_new_class_declarations():
    registry = _registry()
    assert len(registry["declarations"]) == 3
    assert all("kif" not in entry and "cnl" not in entry for entry in registry["declarations"])
    assert registry["existing_parent_pool"] == ["Process", "IntelligenceProduct"]


def test_declaration_requires_matching_source_evidence():
    registry = _registry()
    assert match_declaration(
        "Biometrics is the process",
        "Biometrics is the process of recognizing an individual.",
        registry,
    )["symbol"] == "BiometricsProcess"
    assert match_declaration("Biometrics is the process", "Biometrics is a characteristic.", registry) is None


def test_registry_rejects_wrong_source_packet():
    with pytest.raises(ValueError, match="not tied"):
        load_declaration_registry(SOURCE / "provisional_declarations.json", source_packet_sha256="0" * 64)


def test_source_only_proposal_names_the_stated_sense_and_skips_existing_classes():
    from src.ontology.vocab import load_closed_class_terms
    from src.preprocessing.declarations import parent_candidates, propose_declaration

    known = load_closed_class_terms()
    entry = propose_declaration("Biometrics is the process", known)
    assert entry["symbol"] == "BiometricsProcess"
    assert entry["evidence_cue"] == "is the process"
    assert not any(key in entry for key in ("parent", "subclass", "kif", "cnl"))
    # OpenSourceIntelligence already exists in another sense; do not redeclare it.
    assert propose_declaration("Open-source intelligence is intelligence", known) is None
    assert propose_declaration("The intelligence process is a model", known) is None
    team = propose_declaration("A fusion cell is a team", known)
    assert team["alias"] == "fusion cell"
    # The proposer drops the article, so matching must accept it on the candidate.
    registry = {"declarations": [team]}
    assert match_declaration("A fusion cell is a team", "A fusion cell is a team of analysts.", registry) == team
    assert parent_candidates(entry, ["Process", "IntelligenceProduct"], known) == ["Process", "IntelligenceProduct"]


def test_proposed_registry_must_be_tied_to_its_unlabeled_sources(tmp_path):
    from src.preprocessing.declarations import PROPOSED_STATUS

    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "status": PROPOSED_STATUS, "sources_sha256": "a" * 64,
        "existing_parent_pool": ["Process"], "declarations": [],
    }), encoding="utf-8")
    assert load_declaration_registry(path, sources_sha256="a" * 64)["declarations"] == []
    with pytest.raises(ValueError, match="not tied"):
        load_declaration_registry(path, sources_sha256="b" * 64)
    with pytest.raises(ValueError, match="not tied"):
        load_declaration_registry(path, source_packet_sha256="a" * 64)
