import json

import pytest

from scripts.research_preflight import assess
from src.compiler.compiler import CNLCompiler
from src.eval.comparison_predictors import RulesPredictor
from src.ingest.grounding import DoctrineGrounder


SOURCE = "Human intelligence is the collection by a trained human intelligence collector of foreign information."


def test_background_parent_requires_explicit_opt_in():
    assert RulesPredictor()(SOURCE) is None
    assert RulesPredictor(allow_background_axioms=True)(SOURCE) == "HumanIntelligence subclass-of IntelligenceDiscipline"
    arguments = dict(cnl="HumanIntelligence subclass-of IntelligenceDiscipline",
                     original_text=SOURCE, normalized_text=SOURCE, linked_text=SOURCE, subclaim_text=SOURCE)
    assert not DoctrineGrounder().assess(**arguments).accepted
    assert DoctrineGrounder(allow_background_axioms=True).assess(**arguments).accepted


def test_source_only_keeps_vocabulary_and_explicit_parent():
    source = "Human intelligence is an intelligence discipline."
    assert DoctrineGrounder().assess(
        cnl="HumanIntelligence subclass-of IntelligenceDiscipline", original_text=source,
        normalized_text=source, linked_text=source, subclaim_text=source).accepted


@pytest.mark.parametrize("cnl", [
    "?x is-a Human",
    "some ?x is-a Human agent ?x Human",
    "every ?x is-a Human implies agent ?y ?x",
    "if ?x is-a Human then agent ?x Human",
])
def test_closed_formula_rejects_free_variables(cnl):
    with pytest.raises(ValueError, match="Unbound variables"):
        CNLCompiler().compile(cnl, require_closed=True)


def test_quantifier_scope_preserves_compilation():
    compiler = CNLCompiler()
    cnl = "every ?x is-a Human implies agent ?x Human"
    assert compiler.compile(cnl, require_closed=True) == compiler.compile(cnl)


def test_draft_novel_target_cannot_unlock_preflight(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"record_id": "draft", "doc_id": "test",
        "nl": "A weapon is an artifact.", "status": "draft", "formalizable": True,
        "cnl": "Weapon subclass-of Artifact", "kif": "(subclass Weapon Artifact)"}) + "\n")
    report = assess([benchmark], [])
    assert report["status"] == "blocked"
    assert report["counts"]["beyond_direct_restatements"] == 0
    assert len(report["blockers"]) == 3


def test_preflight_pass_and_training_overlap(tmp_path):
    positive = {"record_id": "positive", "doc_id": "reserved", "status": "reviewed",
                "nl": "Counterintelligence capabilities consist of CI teams.",
                "formalizable": True, "cnl": "CITeam subclass-of CICapability",
                "kif": "(subclass CITeam CICapability)"}
    negative = {"record_id": "negative", "doc_id": "reserved", "status": "reviewed",
                "nl": "An unsupported passage.", "formalizable": False, "cnl": None, "kif": None}
    benchmark, training = tmp_path / "benchmark.jsonl", tmp_path / "training.jsonl"
    benchmark.write_text("\n".join(json.dumps(r) for r in [positive, negative]))
    training.write_text(json.dumps({"nl": "Unrelated training source."}))
    assert assess([benchmark], [training])["status"] == "data_checks_passed"
    training.write_text(json.dumps({"nl": positive["nl"]}))
    report = assess([benchmark], [training])
    assert report["status"] == "blocked"
    assert "positive" in report["training_text_overlap"]
    assert report["counts"]["beyond_direct_restatements"] == 0
