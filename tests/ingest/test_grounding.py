from __future__ import annotations

import json
from pathlib import Path
from src.ingest.grounding import DoctrineGrounder, extract_cnl_terms


def _make_term_files(tmp_root: Path) -> tuple[Path, Path]:
    classes_path = tmp_root / "classes.jsonl"
    relations_path = tmp_root / "relations.jsonl"
    classes = ["MilitaryProcess", "AutonomousAgent", "Transportation", "Region"]
    relations = ["agent", "destination"]
    classes_path.write_text("".join(json.dumps({"term": term}) + "\n" for term in classes), encoding="utf-8")
    relations_path.write_text("".join(json.dumps({"term": term}) + "\n" for term in relations), encoding="utf-8")
    return classes_path, relations_path


def test_extract_cnl_terms_separates_relation_and_class_terms():
    relation_terms, class_terms, all_terms = extract_cnl_terms("agent MilitaryProcess AutonomousAgent")

    assert relation_terms == ["agent"]
    assert class_terms == ["MilitaryProcess", "AutonomousAgent"]
    assert all_terms == ["agent", "MilitaryProcess", "AutonomousAgent"]


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
