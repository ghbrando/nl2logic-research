from __future__ import annotations

import json
from pathlib import Path

from src.ingest.grounding import DoctrineGrounder, extract_cnl_terms


def _make_term_files(tmp_root: Path) -> tuple[Path, Path]:
    classes_path = tmp_root / "classes.jsonl"
    relations_path = tmp_root / "relations.jsonl"
    classes = [
        "MilitaryProcess", "AutonomousAgent", "Transportation", "Region",
        "Likely", "Planning", "Army",
        # Terms needed for weak-argument regression tests
        "Dangerous", "May", "On", "UnitedStates", "Power", "Plan", "Enemy",
        "Security", "Key", "Human", "Surprise",
    ]
    relations = [
        "agent", "destination", "complexity",
        # Relations needed for weak-argument regression tests
        "most", "needs", "time", "states", "during", "domain", "enemy",
        "prevents", "larger", "depth",
    ]
    classes_path.write_text("".join(json.dumps({"term": term}) + "\n" for term in classes), encoding="utf-8")
    relations_path.write_text("".join(json.dumps({"term": term}) + "\n" for term in relations), encoding="utf-8")
    return classes_path, relations_path


def test_extract_cnl_terms_separates_relation_and_class_terms():
    relation_terms, class_terms, all_terms = extract_cnl_terms("agent MilitaryProcess AutonomousAgent")

    assert relation_terms == ["agent"]
    assert class_terms == ["MilitaryProcess", "AutonomousAgent"]
    assert all_terms == ["agent", "MilitaryProcess", "AutonomousAgent"]


def test_extract_cnl_terms_ignores_variable_tokens():
    relation_terms, class_terms, all_terms = extract_cnl_terms("some ?x is-a Army")

    assert relation_terms == []
    assert class_terms == ["Army"]
    assert all_terms == ["Army"]


def test_grounder_accepts_when_relation_and_classes_are_lexically_grounded(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="agent MilitaryProcess AutonomousAgent",
        original_text="A military process has an autonomous agent.",
        normalized_text="A military process has an autonomous agent.",
        linked_text="A military process has an autonomous agent.",
        subclaim_text="A military process has an autonomous agent.",
    )

    assert assessment.accepted is True
    assert assessment.reason is None
    assert assessment.ungrounded_terms == []


def test_grounder_routes_record_to_review_when_class_terms_are_ungrounded(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="agent MilitaryProcess AutonomousAgent",
        original_text="A military process has an agent.",
        normalized_text="A military process has an agent.",
        linked_text="A military process has an agent.",
        subclaim_text="A military process has an agent.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "ungrounded_class_terms"
    assert assessment.ungrounded_terms == ["AutonomousAgent"]


def test_grounder_uses_prompt_aliases_for_known_class_phrases(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="destination Transportation Region",
        original_text="A transportation process has a region as its destination.",
        normalized_text="A transportation process has a region as its destination.",
        linked_text="A transportation process has a region as its destination.",
        subclaim_text="A transportation process has a region as its destination.",
    )

    assert assessment.accepted is True
    assert assessment.reason is None


def test_grounder_rejects_existential_output_without_existential_source_support(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="some ?x is-a Likely",
        original_text="Is most likely to do the most likely threat course of action.",
        normalized_text="Is most likely to do the most likely threat course of action.",
        linked_text="Is most likely to do the most likely threat course of action.",
        subclaim_text="Is most likely to do the most likely threat course of action.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_grounding"
    assert "Existential output" in assessment.detail


def test_grounder_rejects_existential_output_when_some_modifies_non_class_phrase(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="some ?x is-a Army",
        original_text="However, due to the unique characteristics of Army operations, the Army intelligence process steps differ in some important but subtle ways.",
        normalized_text="However, due to the unique characteristics of Army operations, the Army intelligence process steps differ in some important but subtle ways.",
        linked_text="However, due to the unique characteristics of Army operations, the Army intelligence process steps differ in some important but subtle ways.",
        subclaim_text="However, due to the unique characteristics of Army operations, the Army intelligence process steps differ in some important but subtle ways.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_grounding"
    assert "Existential output" in assessment.detail


def test_grounder_rejects_instance_output_without_copular_support(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="?x is-a Planning",
        original_text="Integrates the planning.",
        normalized_text="Integrates the planning.",
        linked_text="Integrates the planning.",
        subclaim_text="Integrates the planning.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_grounding"
    assert "Unary instance output" in assessment.detail


def test_grounder_rejects_self_subclass_form(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="Army subclass-of Army",
        original_text="Army intelligence processes.",
        normalized_text="Army intelligence processes.",
        linked_text="Army intelligence processes.",
        subclaim_text="Army intelligence processes.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "degenerate_form"
    assert "self-subclass" in assessment.detail


def test_grounder_rejects_relation_outputs_with_only_variable_arguments(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="complexity ?x ?y",
        original_text="Complexity to answer the requirements.",
        normalized_text="Complexity to answer the requirements.",
        linked_text="Complexity to answer the requirements.",
        subclaim_text="Complexity to answer the requirements.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "degenerate_form"
    assert "no grounded ontology arguments" in assessment.detail


# ---------------------------------------------------------------------------
# Regression tests: weak single-word argument filter
# These outputs were accepted by the Spark pipeline but are semantically
# meaningless and must be rejected (routed to review).
# ---------------------------------------------------------------------------

def test_rejects_most_dangerous(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="most Dangerous ?x",
        original_text="Is most likely to do the most dangerous threat course of action.",
        normalized_text="Is most likely to do the most dangerous threat course of action.",
        linked_text="Is most likely to do the most dangerous threat course of action.",
        subclaim_text="Is most likely to do the most dangerous threat course of action.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "Dangerous" in assessment.detail


def test_rejects_needs_may(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="needs ?x May",
        original_text="The commander may need additional information.",
        normalized_text="The commander may need additional information.",
        linked_text="The commander may need additional information.",
        subclaim_text="The commander may need additional information.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "May" in assessment.detail


def test_rejects_time_on(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="time ?x On",
        original_text="Focus on the time available for planning.",
        normalized_text="Focus on the time available for planning.",
        linked_text="Focus on the time available for planning.",
        subclaim_text="Focus on the time available for planning.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "On" in assessment.detail


def test_rejects_states_unitedstates(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="states ?x UnitedStates",
        original_text="The United States Army states its doctrine.",
        normalized_text="The United States Army states its doctrine.",
        linked_text="The United States Army states its doctrine.",
        subclaim_text="The United States Army states its doctrine.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "UnitedStates" in assessment.detail


def test_rejects_during_power(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="during ?x Power",
        original_text="During the exercise of power projection.",
        normalized_text="During the exercise of power projection.",
        linked_text="During the exercise of power projection.",
        subclaim_text="During the exercise of power projection.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "Power" in assessment.detail


def test_rejects_domain_plan(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="domain ?x Plan",
        original_text="The plan is part of the intelligence domain.",
        normalized_text="The plan is part of the intelligence domain.",
        linked_text="The plan is part of the intelligence domain.",
        subclaim_text="The plan is part of the intelligence domain.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "Plan" in assessment.detail


def test_rejects_enemy_enemy(tmp_path: Path):
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="enemy ?x Enemy",
        original_text="The enemy force is the primary enemy threat.",
        normalized_text="The enemy force is the primary enemy threat.",
        linked_text="The enemy force is the primary enemy threat.",
        subclaim_text="The enemy force is the primary enemy threat.",
    )

    assert assessment.accepted is False
    assert assessment.reason == "weak_argument"
    assert "Enemy" in assessment.detail


def test_legitimate_relation_not_blocked_by_weak_argument_filter(tmp_path: Path):
    """Ensure multi-word PascalCase arguments and specific terms still pass."""
    classes_path, relations_path = _make_term_files(tmp_path)
    grounder = DoctrineGrounder(classes_path=classes_path, relations_path=relations_path)

    assessment = grounder.assess(
        cnl="agent MilitaryProcess AutonomousAgent",
        original_text="A military process has an autonomous agent.",
        normalized_text="A military process has an autonomous agent.",
        linked_text="A military process has an autonomous agent.",
        subclaim_text="A military process has an autonomous agent.",
    )

    assert assessment.accepted is True
