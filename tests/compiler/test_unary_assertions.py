import json
import pytest
from pathlib import Path

SUMO_CLASSES = Path("data/training_pairs/sumo_classes.jsonl")


def _load_classes() -> list[dict]:
    with open(SUMO_CLASSES, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _instance_pairs() -> list[tuple[str, str]]:
    pairs = []
    for c in _load_classes():
        term = c["term"]
        pairs.append((f"?x is-a {term}", f"(instance ?x {term})"))
        if len(pairs) >= 20:
            break
    return pairs


def _subclass_pairs() -> list[tuple[str, str]]:
    terms = [c["term"] for c in _load_classes()[:20]]
    pairs = []
    for i in range(0, len(terms) - 1, 2):
        child, parent = terms[i], terms[i + 1]
        pairs.append((f"{child} subclass-of {parent}", f"(subclass {child} {parent})"))
    return pairs


# ── Tests ──────────────────────────────────────────────────────────────────

class TestInstanceAssertions:

    def test_basic_instance(self, compiler):
        assert compiler.compile("?x is-a Process") == "(instance ?x Process)"

    def test_instance_military_process(self, compiler):
        assert compiler.compile("?x is-a MilitaryProcess") == "(instance ?x MilitaryProcess)"

    def test_instance_named_var(self, compiler):
        assert compiler.compile("?attack is-a Process") == "(instance ?attack Process)"

    @pytest.mark.parametrize("cnl,expected", _instance_pairs())
    def test_instance_sumo_classes(self, compiler, cnl, expected):
        assert compiler.compile(cnl) == expected


class TestSubclassAssertions:

    def test_basic_subclass(self, compiler):
        assert compiler.compile("Process subclass-of Entity") == "(subclass Process Entity)"

    def test_subclass_military(self, compiler):
        assert compiler.compile("MilitaryUnit subclass-of Entity") == "(subclass MilitaryUnit Entity)"

    @pytest.mark.parametrize("cnl,expected", _subclass_pairs())
    def test_subclass_sumo_classes(self, compiler, cnl, expected):
        assert compiler.compile(cnl) == expected


class TestDoctrineExtensionTerms:
    """Doctrine domain extension terms must be accepted by the compiler.

    These terms are defined in data/ontology/doctrine_domain.kif and are NOT
    in sumo_classes.jsonl.  The compiler must accept them after the shared
    vocab loader unions in doctrine_domain.kif.
    """

    @pytest.mark.parametrize("cnl,expected", [
        (
            "CombatInformation subclass-of FactualText",
            "(subclass CombatInformation FactualText)",
        ),
        (
            "IntelligenceProcess subclass-of MilitaryProcess",
            "(subclass IntelligenceProcess MilitaryProcess)",
        ),
        (
            "IntelligenceEnterprise subclass-of MilitaryOrganization",
            "(subclass IntelligenceEnterprise MilitaryOrganization)",
        ),
        (
            "IntelligenceProfessional subclass-of MilitaryPerson",
            "(subclass IntelligenceProfessional MilitaryPerson)",
        ),
        (
            "IntelligenceWarfightingFunction subclass-of WarfightingFunction",
            "(subclass IntelligenceWarfightingFunction WarfightingFunction)",
        ),
        (
            "?x is-a CombatInformation",
            "(instance ?x CombatInformation)",
        ),
        (
            "?x is-a IntelligenceProcess",
            "(instance ?x IntelligenceProcess)",
        ),
    ])
    def test_doctrine_term_compiles(self, compiler, cnl, expected):
        assert compiler.compile(cnl) == expected