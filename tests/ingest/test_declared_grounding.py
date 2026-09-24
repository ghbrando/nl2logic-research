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
