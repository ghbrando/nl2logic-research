"""Tests for src.ontology.vocab — shared closed-vocabulary source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ontology.vocab import (
    extract_kif_class_terms,
    load_closed_class_terms,
    load_closed_relation_terms,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# extract_kif_class_terms
# ---------------------------------------------------------------------------

class TestExtractKifClassTerms:

    def test_finds_subclass_child_names(self, tmp_path: Path):
        kif = tmp_path / "test.kif"
        kif.write_text(
            "(subclass CombatInformation FactualText)\n"
            "(subclass IntelligenceProcess MilitaryProcess)\n",
            encoding="utf-8",
        )
        terms = extract_kif_class_terms(kif)
        assert "CombatInformation" in terms
        assert "IntelligenceProcess" in terms

    def test_does_not_include_parent_class(self, tmp_path: Path):
        kif = tmp_path / "test.kif"
        kif.write_text("(subclass Child Parent)\n", encoding="utf-8")
        terms = extract_kif_class_terms(kif)
        assert "Child" in terms
        assert "Parent" not in terms

    def test_ignores_documentation_lines(self, tmp_path: Path):
        kif = tmp_path / "test.kif"
        kif.write_text(
            "(subclass Foo Bar)\n"
            '(documentation Foo EnglishLanguage "A class.")\n',
            encoding="utf-8",
        )
        terms = extract_kif_class_terms(kif)
        assert terms == {"Foo"}

    def test_ignores_comment_lines(self, tmp_path: Path):
        kif = tmp_path / "test.kif"
        kif.write_text(
            ";; This is a comment\n"
            "(subclass Alpha Beta)\n",
            encoding="utf-8",
        )
        terms = extract_kif_class_terms(kif)
        assert terms == {"Alpha"}

    def test_empty_file_returns_empty_set(self, tmp_path: Path):
        kif = tmp_path / "empty.kif"
        kif.write_text("", encoding="utf-8")
        assert extract_kif_class_terms(kif) == set()


# ---------------------------------------------------------------------------
# load_closed_class_terms
# ---------------------------------------------------------------------------

class TestLoadClosedClassTerms:

    def test_includes_sumo_class_terms(self, tmp_path: Path):
        sumo = tmp_path / "classes.jsonl"
        _write_jsonl(sumo, [{"term": "Report"}, {"term": "MilitaryProcess"}])
        terms = load_closed_class_terms(sumo, doctrine_kif=None)
        assert "Report" in terms
        assert "MilitaryProcess" in terms

    def test_includes_doctrine_extension_terms(self, tmp_path: Path):
        sumo = tmp_path / "classes.jsonl"
        _write_jsonl(sumo, [{"term": "FactualText"}])
        doctrine = tmp_path / "domain.kif"
        doctrine.write_text(
            "(subclass CombatInformation FactualText)\n",
            encoding="utf-8",
        )
        terms = load_closed_class_terms(sumo, doctrine)
        assert "FactualText" in terms
        assert "CombatInformation" in terms

    def test_doctrine_kif_none_uses_only_sumo(self, tmp_path: Path):
        sumo = tmp_path / "classes.jsonl"
        _write_jsonl(sumo, [{"term": "Report"}])
        terms = load_closed_class_terms(sumo, doctrine_kif=None)
        assert "Report" in terms
        assert "CombatInformation" not in terms

    def test_missing_doctrine_kif_is_handled_gracefully(self, tmp_path: Path):
        sumo = tmp_path / "classes.jsonl"
        _write_jsonl(sumo, [{"term": "Report"}])
        missing = tmp_path / "nonexistent.kif"
        terms = load_closed_class_terms(sumo, missing)
        assert "Report" in terms  # SUMO terms still returned

    def test_real_doctrine_terms_in_closed_vocabulary(self):
        """Integration: doctrine_domain.kif terms appear in the closed vocabulary."""
        terms = load_closed_class_terms()
        for expected in [
            "CombatInformation",
            "IntelligenceProcess",
            "IntelligenceEnterprise",
            "IntelligenceProfessional",
            "IntelligenceWarfightingFunction",
            "WarfightingFunction",
            "TacticalCommander",
        ]:
            assert expected in terms, f"{expected} missing from closed vocabulary"

    def test_real_sumo_terms_still_present(self):
        """Integration: base SUMO terms are not lost after doctrine union."""
        terms = load_closed_class_terms()
        for expected in ["Report", "FactualText", "MilitaryProcess", "MilitaryPerson"]:
            assert expected in terms, f"{expected} missing from closed vocabulary"


# ---------------------------------------------------------------------------
# load_closed_relation_terms
# ---------------------------------------------------------------------------

class TestLoadClosedRelationTerms:

    def test_includes_sumo_relation_terms_with_arity(self, tmp_path: Path):
        rels = tmp_path / "relations.jsonl"
        _write_jsonl(rels, [{"term": "agent", "arity": 2}, {"term": "result"}])
        terms = load_closed_relation_terms(rels, doctrine_kif=None)
        assert "agent" in terms
        assert terms["agent"] == 2
        assert "result" in terms

    def test_doctrine_kif_param_accepted_but_no_relations_added(self, tmp_path: Path):
        """doctrine_domain.kif defines no new relations; param accepted for future use."""
        rels = tmp_path / "relations.jsonl"
        _write_jsonl(rels, [{"term": "agent"}])
        doctrine = tmp_path / "domain.kif"
        doctrine.write_text("(subclass Foo Bar)\n", encoding="utf-8")
        terms = load_closed_relation_terms(rels, doctrine)
        assert list(terms.keys()) == ["agent"]
