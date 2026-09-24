import pytest

from src.ingest.grounding import assess_declared_definition

PARENTS = ("Process", "IntelligenceProduct")
BIOMETRICS = {
    "alias": "Biometrics", "evidence_cue": "is the process of",
    "symbol": "BiometricsProcess", "declaration": "(instance BiometricsProcess Class)",
}
OSINT = {
    "alias": "Open-source intelligence", "evidence_cue": "is intelligence that is produced",
    "symbol": "OpenSourceIntelligenceProduct",
}
TECHINT = {
    "alias": "Technical intelligence", "evidence_cue": "is intelligence derived from",
    "symbol": "TechnicalIntelligenceProduct",
}
BIO_SENTENCE = ("Biometrics is the process of recognizing an individual based on measurable "
                "anatomical, physiological, and behavioral characteristics (JP 2-0).")
OSINT_SENTENCE = ("Open-source intelligence is intelligence that is produced from publicly "
                  "available information and is collected, exploited, and disseminated.")
TECHINT_SENTENCE = "Technical intelligence is intelligence derived from the collection of data."


def _assess(cnl, sentence, declaration, passage=None):
    passage = passage if passage is not None else f"Other context may apply. {sentence} More text."
    return assess_declared_definition(
        cnl, passage=passage, evidence_sentence=sentence,
        declaration=declaration, parent_terms=PARENTS,
    )


def test_passage_stated_genus_is_accepted_despite_modality_elsewhere():
    result = _assess("BiometricsProcess subclass-of Process", BIO_SENTENCE, BIOMETRICS)
    assert result.accepted, result.detail


def test_product_sense_requires_passage_local_production_cue():
    assert _assess("OpenSourceIntelligenceProduct subclass-of IntelligenceProduct",
                   OSINT_SENTENCE, OSINT).accepted
    ambiguous = _assess("TechnicalIntelligenceProduct subclass-of IntelligenceProduct",
                        TECHINT_SENTENCE, TECHINT)
    assert (ambiguous.accepted, ambiguous.reason) == (False, "ambiguous_genus_sense")


@pytest.mark.parametrize("cnl, sentence, declaration, reason", [
    ("Source subclass-of IntelligenceProduct", OSINT_SENTENCE, OSINT, "undeclared_child"),
    ("Source subclass-of OpenSourceIntelligenceProduct", OSINT_SENTENCE, OSINT, "reversed_direction"),
    ("Process subclass-of TechnicalIntelligenceProduct", TECHINT_SENTENCE, TECHINT, "reversed_direction"),
    ("TechnicalIntelligenceProduct subclass-of Process", TECHINT_SENTENCE, TECHINT, "unstated_parent"),
    ("BiometricsProcess subclass-of IntelligenceProduct", BIO_SENTENCE, BIOMETRICS, "unstated_parent"),
    ("BiometricsProcess subclass-of Entity", BIO_SENTENCE, BIOMETRICS, "parent_not_existing_class"),
    ("?x is-a BiometricsProcess", BIO_SENTENCE, BIOMETRICS, "not_single_subclass"),
])
def test_incidental_reversed_and_unstated_claims_are_rejected(cnl, sentence, declaration, reason):
    result = _assess(cnl, sentence, declaration)
    assert (result.accepted, result.reason) == (False, reason)


def test_declaration_is_not_proof_without_current_passage_support():
    # The declared cue alone, or a sentence supplied from memory, is not evidence.
    result = _assess("BiometricsProcess subclass-of Process", BIO_SENTENCE, BIOMETRICS,
                     passage="Biometric characteristics can be measured.")
    assert result.reason == "evidence_not_in_passage"
    other_sense = "Biometrics is a discipline that studies the process of recognition."
    result = _assess("BiometricsProcess subclass-of Process", other_sense, BIOMETRICS)
    assert result.reason == "declaration_sense_not_in_passage"


def test_declared_phrase_must_be_the_definiendum():
    sentence = "Analysts use biometrics is the process of matching records."
    result = _assess("BiometricsProcess subclass-of Process", sentence, BIOMETRICS)
    assert result.reason == "declaration_phrase_not_subject"


def test_scope_inside_the_evidence_sentence_is_not_accepted():
    sentence = "Biometrics is the process of recognizing an individual who may be hostile."
    result = _assess("BiometricsProcess subclass-of Process", sentence, BIOMETRICS)
    assert result.reason == "unverified_scope"


def _declared(alias, cue, symbol, sentence, cnl, parents=("Process", "IntelligenceProduct")):
    return assess_declared_definition(
        cnl, passage=sentence, evidence_sentence=sentence,
        declaration={"alias": alias, "evidence_cue": cue, "symbol": symbol}, parent_terms=parents,
    )


# Synthetic frames; they reproduce the failure shapes seen on chapter 3+ without
# reusing those sentences, so the fix is not fitted to the queried passages.
@pytest.mark.parametrize("sentence, cue, parent", [
    ("The ledger is a key record of unit expenditures.", "is a key", "Key"),
    ("A relay post is the Army's forward signal node.", "is the Army", "Army"),
    ("The drill design is organic to each squadron.", "is organic", "Organic"),
    ("The two support methods are area coverage and point coverage.", "are area", "Area"),
    ("A watch cell is a process team within the staff.", "is a process", "Process"),
])
def test_modifier_possessive_and_adjective_heads_are_not_genus(sentence, cue, parent):
    alias = sentence.split(" is ")[0].split(" are ")[0].removeprefix("The ").removeprefix("A ")
    result = _declared(alias, cue, "NewThing", sentence, f"NewThing subclass-of {parent}",
                       parents=("Process", "IntelligenceProduct", parent))
    assert (result.accepted, result.reason) == (False, "unstated_parent")


def test_reviewed_genus_head_may_carry_modifiers_but_not_coordination():
    accepted = _declared("Risk review", "is the unit", "RiskReviewProcess",
                         "Risk review is the unit's primary process for weighing hazards.",
                         "RiskReviewProcess subclass-of Process")
    assert accepted.accepted, accepted.detail
    coordinated = _declared("Liaison", "is a process", "LiaisonProcess",
                            "Liaison is a process and a product of coordination.",
                            "LiaisonProcess subclass-of Process")
    assert coordinated.reason == "unstated_parent"


def test_unreviewed_parent_must_be_the_whole_article_introduced_genus():
    whole = _declared("Rally cycle", "is a cycle", "RallyCycle",
                      "A rally cycle is a cycle of regrouping used by convoys.",
                      "RallyCycle subclass-of Cycle", parents=("Process", "Cycle"))
    assert whole.accepted, whole.detail
    modified = _declared("Rally cycle", "is a daily", "RallyCycle",
                         "A rally cycle is a daily cycle of regrouping.",
                         "RallyCycle subclass-of Cycle", parents=("Process", "Cycle"))
    assert modified.reason == "unstated_parent"


def test_symbol_may_not_import_a_sense_the_passage_does_not_state():
    result = _declared("Risk review", "is the unit", "RiskReviewUnit",
                       "Risk review is the unit's primary process for weighing hazards.",
                       "RiskReviewUnit subclass-of Process")
    assert result.reason == "symbol_sense_mismatch"
