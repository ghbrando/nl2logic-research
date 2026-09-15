"""The consist-of enumeration rule.

Doctrine states a category by listing its members ("CI capabilities consist of
CI teams and assigned biometric collection equipment"). These tests pin the
categorical reading recorded in data/ontology/doctrine_domain.kif, and pin the
refusals that keep it from over-reaching.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.compiler.compiler import CNLCompiler
from src.eval.comparison_predictors import RulesPredictor
from src.fsm.cnl_fsm import _singularize_phrase
from src.ingest.grounding import DoctrineGrounder, _normalise_text

CANDIDATES = Path("data/benchmarks/fm2-0/chapter1-2023-review/candidates.jsonl")


@pytest.fixture(scope="module")
def doctrine_text() -> dict[str, str]:
    return {
        json.loads(line)["record_id"]: json.loads(line)["nl"]
        for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


@pytest.fixture(scope="module")
def rules() -> RulesPredictor:
    return RulesPredictor()


def _assess(cnl: str, source: str):
    normalized = _normalise_text(source)
    return DoctrineGrounder().assess(
        cnl=cnl, original_text=source, normalized_text=normalized,
        linked_text=normalized, subclaim_text=normalized,
    )


# --- singularization -------------------------------------------------------

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("CI teams", "CI team"),
        ("HUMINT operations cells", "HUMINT operations cell"),  # only the head noun
        ("CI capabilities", "CI capability"),
        ("assigned biometric collection equipment", "assigned biometric collection equipment"),
        ("analysis", "analysis"),
    ],
)
def test_singularize_touches_only_the_head_noun(phrase: str, expected: str) -> None:
    assert _singularize_phrase(phrase) == expected


# --- the rule itself -------------------------------------------------------

def test_counterintelligence_enumeration(rules: RulesPredictor, doctrine_text: dict[str, str]) -> None:
    cnl = rules(doctrine_text["fm2-0-2023-p33-1-67"])
    assert cnl == (
        "CITeam subclass-of CICapability "
        "BiometricCollectionEquipment subclass-of CICapability"
    )


def test_humint_enumeration_resolves_all_three_members(
    rules: RulesPredictor, doctrine_text: dict[str, str]
) -> None:
    cnl = rules(doctrine_text["fm2-0-2023-p34-1-73"])
    assert cnl == (
        "HUMINTCollectionTeam subclass-of HUMINTCapability "
        "HUMINTOperationsCell subclass-of HUMINTCapability "
        "BiometricCollectionEquipment subclass-of HUMINTCapability"
    )


def test_operations_cells_does_not_collapse_to_the_biology_class(
    rules: RulesPredictor, doctrine_text: dict[str, str]
) -> None:
    """"HUMINT operations cells" must not resolve to SUMO's Cell."""
    cnl = rules(doctrine_text["fm2-0-2023-p34-1-73"])
    children = [claim.split()[0] for claim in cnl.split(" subclass-of ")[:-1]]
    assert "Cell" not in [child.split()[-1] for child in children]
    assert "HUMINTOperationsCell subclass-of HUMINTCapability" in cnl


@pytest.mark.parametrize(
    "record_id",
    [
        "fm2-0-2023-p33-1-70",  # "manned and unmanned platforms and aerial and space-based ..."
        "fm2-0-2023-p34-1-77",  # "chemical, biological, radiological, and nuclear (CBRN) detectors"
        "fm2-0-2023-p36-1-86",  # "theater army collection and survey systems"
    ],
)
def test_unsegmentable_enumerations_abstain(
    rules: RulesPredictor, doctrine_text: dict[str, str], record_id: str
) -> None:
    """A coordinated noun phrase does not split into items, so the rule refuses.

    Emitting the members it can resolve would state less than the source does.
    """
    assert rules(doctrine_text[record_id]) is None


def test_single_member_enumeration_is_refused(rules: RulesPredictor) -> None:
    assert rules("CI capabilities consist of CI teams.") is None


def test_unknown_member_refuses_the_whole_enumeration(rules: RulesPredictor) -> None:
    assert rules("CI capabilities consist of CI teams and quantum widgets.") is None


def test_unknown_category_is_refused(rules: RulesPredictor) -> None:
    assert rules("Widget capabilities consist of CI teams and assigned biometric collection equipment.") is None


# --- compiler and gate -----------------------------------------------------

def test_enumeration_compiles_to_one_kif_assertion_per_member(
    rules: RulesPredictor, doctrine_text: dict[str, str]
) -> None:
    cnl = rules(doctrine_text["fm2-0-2023-p33-1-67"])
    kif = CNLCompiler().compile(cnl)
    assert kif.splitlines() == [
        "(subclass CITeam CICapability)",
        "(subclass BiometricCollectionEquipment CICapability)",
    ]


@pytest.mark.parametrize(
    "record_id", ["fm2-0-2023-p33-1-67", "fm2-0-2023-p34-1-73"],
)
def test_enumeration_output_is_accepted_by_the_grounding_gate(
    rules: RulesPredictor, doctrine_text: dict[str, str], record_id: str
) -> None:
    source = doctrine_text[record_id]
    result = _assess(rules(source), source)
    assert result.accepted, result.reason


def test_reversed_enumeration_claims_are_refused(doctrine_text: dict[str, str]) -> None:
    source = doctrine_text["fm2-0-2023-p33-1-67"]
    result = _assess(
        "CICapability subclass-of CITeam "
        "CICapability subclass-of BiometricCollectionEquipment",
        source,
    )
    assert not result.accepted
    assert result.reason == "unverified_subclass_direction"


def test_one_bad_claim_refuses_the_whole_output(doctrine_text: dict[str, str]) -> None:
    """A multi-claim output must not pass because most of its claims are sound."""
    source = doctrine_text["fm2-0-2023-p33-1-67"]
    result = _assess(
        "CITeam subclass-of CICapability "
        "CICapability subclass-of BiometricCollectionEquipment",
        source,
    )
    assert not result.accepted
    assert result.reason == "unverified_subclass_direction"


def test_degenerate_claim_inside_an_enumeration_is_caught(doctrine_text: dict[str, str]) -> None:
    source = doctrine_text["fm2-0-2023-p33-1-67"]
    result = _assess(
        "CITeam subclass-of CICapability CICapability subclass-of CICapability",
        source,
    )
    assert not result.accepted
    assert result.reason == "degenerate_form"


def test_doctrine_ontology_does_not_assert_the_claims_it_should_derive() -> None:
    """Asserting the enumeration in the ontology would make extraction circular."""
    kif = Path("data/ontology/doctrine_domain.kif").read_text(encoding="utf-8")
    for claim in (
        "(subclass CITeam CICapability)",
        "(subclass BiometricCollectionEquipment CICapability)",
        "(subclass HUMINTCollectionTeam HUMINTCapability)",
        "(subclass HUMINTOperationsCell HUMINTCapability)",
    ):
        assert claim not in kif, claim
