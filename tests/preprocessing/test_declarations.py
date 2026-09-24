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


def test_source_only_proposal_names_the_genus_head_and_declines_non_kinds():
    from src.ontology.vocab import load_closed_class_terms
    from src.preprocessing.declarations import parent_candidates, propose_declaration

    known = load_closed_class_terms()
    entry = propose_declaration("Biometrics is the process", known,
                                "Biometrics is the process of recognizing an individual.")
    assert (entry["symbol"], entry["evidence_cue"], entry["genus_phrase"]) == ("BiometricsProcess", "is the", "process")
    assert not any(key in entry for key in ("parent", "subclass", "kif", "cnl"))
    assert parent_candidates(entry, ["Process", "IntelligenceProduct"], known) == ["Process", "IntelligenceProduct"]
    # A possessive modifier is not the head; the genus head names the sense.
    assert propose_declaration("Risk review is the unit", known,
                               "Risk review is the unit's primary process for weighing hazards.")["symbol"] == "RiskReviewProcess"
    # OpenSourceIntelligence already exists in another sense; do not redeclare it.
    assert propose_declaration("Open-source intelligence is intelligence", known,
                               "Open-source intelligence is intelligence that is produced.") is None
    for candidate, sentence in [
        ("The ledger is a key", "The ledger is a key component of the budget."),
        ("The drill design is organic", "The drill design is organic to each squadron."),
        ("This drill is a process", "This drill is a process of rehearsal."),
        ("Liaison is a process", "Liaison is a process and a product of coordination."),
    ]:
        assert propose_declaration(candidate, known, sentence) is None, candidate
    team = propose_declaration("A fusion cell is a team", known, "A fusion cell is a team of analysts.")
    assert (team["alias"], team["symbol"]) == ("fusion cell", "FusionCellTeam")
    # The proposer drops the article, so matching must accept it on the candidate.
    registry = {"declarations": [team]}
    assert match_declaration("A fusion cell is a team", "A fusion cell is a team of analysts.", registry) == team


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


def test_proposal_declines_verbs_possessive_fragments_and_plural_non_kinds():
    from src.ontology.vocab import load_closed_class_terms
    from src.preprocessing.declarations import propose_declaration

    known = load_closed_class_terms()
    for candidate, sentence in [
        ("The value of the depot is the ability", "The value of the depot is the ability it provides to planners."),
        ("The following are the five", "The following are the five types of patrols:"),
        ("The squad is the Army", "The squad is the Army's most flexible element."),
        ("Drill cards are the three", "Drill cards are the three types of drills used."),
    ]:
        assert propose_declaration(candidate, known, sentence) is None, candidate
