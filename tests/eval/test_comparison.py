import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.compare_formalizers import main
from scripts.finalize_benchmark_review import finalize
from scripts.prepare_benchmark_review import first_sentence, page_candidates, select_candidates
from src.eval.comparison import evaluate_method, read_benchmark, restates_ontology, summarize, training_overlap
from src.eval.comparison_predictors import RulesPredictor, Seq2SeqPredictor


def row(rid="r1", status="reviewed", formalizable=True):
    return {"record_id": rid, "doc_id": "test", "nl": "A weapon is an artifact.",
            "status": status, "formalizable": formalizable, "split": "eval", "pattern": "subclass",
            "cnl": "Weapon subclass-of Artifact" if formalizable else None,
            "kif": "(subclass Weapon Artifact)" if formalizable else None}


class AcceptingGrounder:
    def assess(self, **kwargs):
        return SimpleNamespace(accepted=True, reason=None)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_scoring_separates_drafts_errors_and_false_accepts():
    rows = [row(), row("r2", formalizable=False), row("r3", status="draft"), row("r4"), row("r5", formalizable=False)]
    predictions = iter(["Weapon subclass-of Artifact"] * 3 + [RuntimeError("broken model"), None])
    inputs = []
    def predict(text):
        inputs.append(text)
        value = next(predictions)
        if isinstance(value, Exception):
            raise value
        return value
    results = evaluate_method(rows, predict, grounder=AcceptingGrounder())
    metrics = summarize(results)
    assert all(isinstance(text, str) for text in inputs)
    assert metrics["scored"] == 4
    assert metrics["accepted_exact_precision"] == .5
    assert metrics["accepted_coverage"] == .5
    assert metrics["correct_accepted_positive_coverage"] == .5
    assert metrics["abstention_accuracy"] == .5
    assert metrics["false_accepts_on_abstain_gold"] == 1
    assert results[2]["kif_exact"] is None
    assert results[3]["route"] == "error"


def test_zero_acceptance_is_not_perfect_precision():
    metrics = summarize(evaluate_method([row()], lambda text: None))
    assert metrics["accepted_exact_precision"] is None
    assert metrics["accepted_coverage"] == 0
    assert summarize([])["accepted_coverage"] is None


def test_shared_grounder_blocks_reversed_claim():
    r = row()
    r["nl"] = "An artifact is a weapon."
    result = evaluate_method([r], lambda text: "Weapon subclass-of Artifact")[0]
    assert result["route"] == "review"
    assert result["reason"] == "unverified_subclass_direction"


def test_compiler_failure_is_not_abstention():
    result = evaluate_method([row(formalizable=False)], lambda text: "between Weapon Artifact")[0]
    assert result["route"] == "compile_error"
    assert summarize([result])["abstention_accuracy"] == 0


def test_unbound_prediction_cannot_be_accepted():
    result = evaluate_method([row()], lambda text: "?x is-a Weapon",
                             grounder=AcceptingGrounder())[0]
    assert result["route"] == "compile_error"
    assert "Unbound variables" in result["error"]
    assert result["kif"] is None


def test_training_overlap_is_unscored(tmp_path):
    path = tmp_path / "training.jsonl"
    write_rows(path, [{"nl": "  A WEAPON  is an artifact. "}])
    overlap = training_overlap([row()], [path])
    results = evaluate_method([row()], lambda text: "Weapon subclass-of Artifact", overlaps=overlap)
    assert results[0]["exclusion"] == "training_text_overlap"
    assert summarize(results)["scored"] == 0


@pytest.mark.parametrize("changes", [
    {"formalizable": None}, {"cnl": None}, {"kif": "(subclass Artifact Weapon)"},
    {"formalizable": False}, {"status": "not_reviewed"},
    {"cnl": "not valid syntax"},
])
def test_malformed_reviewed_gold_is_rejected(tmp_path, changes):
    path = tmp_path / "bad.jsonl"
    write_rows(path, [{**row(), **changes}])
    with pytest.raises(ValueError):
        read_benchmark([path])


def test_unbound_gold_is_rejected(tmp_path):
    path = tmp_path / "unbound.jsonl"
    write_rows(path, [{**row(), "cnl": "?x is-a Weapon", "kif": "(instance ?x Weapon)"}])
    with pytest.raises(ValueError, match="Unbound variables"):
        read_benchmark([path])


def test_draft_labels_are_never_used_as_gold(tmp_path):
    path = tmp_path / "draft.jsonl"
    write_rows(path, [{**row(status="draft"), "cnl": "not valid CNL", "formalizable": None}])
    records = read_benchmark([path])
    assert summarize(evaluate_method(records, lambda text: None))["scored"] == 0


def test_duplicate_ids_fail(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    write_rows(path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        read_benchmark([path])


def test_duplicate_passages_cannot_inflate_scores(tmp_path):
    path = tmp_path / "duplicate-text.jsonl"
    write_rows(path, [row(), row("r2")])
    with pytest.raises(ValueError, match="duplicate source text"):
        read_benchmark([path])


def test_missing_models_report_unavailable_without_oracle(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_rows(benchmark, [row()])
    output = tmp_path / "run"
    assert main(["--benchmark", str(benchmark), "--output-dir", str(output)]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["methods"]["raw"]["metrics"] is None
    assert summary["methods"]["constrained"]["status"] == "unavailable"
    assert not (output / "raw.jsonl").exists()


def test_rule_router_abstains_on_ambiguous_slots():
    predictor = RulesPredictor()
    predictor.router._infer_prompt_plan = lambda _: SimpleNamespace(
        template="subclass", first_terms=("Weapon", "Artifact"), second_terms=("Entity",))
    assert predictor("Anything") is None


def test_constrained_predictor_never_falls_back():
    predictor = Seq2SeqPredictor(None, None, constrained=True)
    def fail(*args, **kwargs):
        raise RuntimeError("constraint failed")
    predictor.sampler.sample = fail
    with pytest.raises(RuntimeError, match="constraint failed"):
        predictor("A weapon is an artifact.")


def test_raw_arm_matches_generation_settings(monkeypatch):
    import sys
    from contextlib import nullcontext
    from src.fsm.cnl_fsm import _DEFAULT_CONSTRAINED_NUM_BEAMS, _DEFAULT_CONSTRAINED_LENGTH_PENALTY
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        return [[1]]
    model = SimpleNamespace(generate=generate)
    tokenizer = SimpleNamespace(batch_decode=lambda *a, **k: ["Weapon subclass-of Artifact"])
    predictor = Seq2SeqPredictor(model, tokenizer, constrained=False, max_tokens=75)
    predictor.sampler._tokenize_prompt = lambda _: {"input_ids": [[1]]}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    assert predictor("A weapon is an artifact.") == "Weapon subclass-of Artifact"
    assert captured["num_beams"] == _DEFAULT_CONSTRAINED_NUM_BEAMS
    assert captured["length_penalty"] == _DEFAULT_CONSTRAINED_LENGTH_PENALTY
    assert captured["max_new_tokens"] == 75
    assert "prefix_allowed_tokens_fn" not in captured


def test_extraction_keeps_pdf_and_printed_pages_separate():
    text = "Chapter 1\n1-7. The U.S. Army provides support to the commander. Another sentence follows.\n1-3 FM 2-0 01 October 2023"
    results = page_candidates(text, pdf_page=19, chapter=1, doc_id="fm2-0-2023")
    assert len(results) == 1
    assert results[0]["nl"] == "The U.S. Army provides support to the commander."
    assert results[0]["pdf_page"] == 19
    assert results[0]["page"] == "1-3"
    assert results[0]["formalizable"] is None
    assert results[0]["status"] == "draft"


def test_selection_is_reproducible_and_excludes_existing_text():
    records = [{"record_id": str(i), "nl": f"Sentence {i}", "phenomena": ["other"], "pdf_page": 1, "paragraph": f"1-{i}"} for i in range(10)]
    first = select_candidates(records, 5, 42, {"sentence 1"})
    assert first == select_candidates(records, 5, 42, {"sentence 1"})
    assert len(first) == 5
    assert all(r["record_id"] != "1" for r in first)


def annotation(**changes):
    return {"record_id": "r1", "nl": row()["nl"], "formalizable": "true",
            "cnl": "Weapon subclass-of Artifact", "pattern": "subclass",
            "status": "reviewed", "reviewed_by": "test-reviewer", **changes}


def test_review_export_requires_explicit_review_and_preserves_source():
    assert finalize([row(status="draft")], [annotation(status="draft")]) == []
    result = finalize([row(status="draft")], [annotation()])[0]
    assert result["kif"] == "(subclass Weapon Artifact)"
    assert result["reviewed_by"] == "test-reviewer"
    assert result["reviewer_type"] == "unspecified"
    ai_result = finalize([row(status="draft")], [annotation(reviewer_type="ai")])[0]
    assert ai_result["reviewer_type"] == "ai"


@pytest.mark.parametrize("changes", [{"reviewed_by": ""}, {"formalizable": ""}, {"nl": "changed"}, {"review_seconds": "nan"}, {"cnl": "not valid syntax"}])
def test_review_export_rejects_incomplete_review(changes):
    with pytest.raises(ValueError):
        finalize([row(status="draft")], [annotation(**changes)])


def ontology_row(rid="o1"):
    """A record whose gold answer the doctrine ontology already asserts."""
    return {
        "record_id": rid, "doc_id": "test",
        "nl": "Human intelligence is intelligence derived from people.",
        "status": "reviewed", "formalizable": True, "split": "eval", "pattern": "subclass",
        "cnl": "HumanIntelligence subclass-of IntelligenceDiscipline",
        "kif": "(subclass HumanIntelligence IntelligenceDiscipline)",
    }


def test_restates_ontology_detects_an_asserted_claim():
    assert restates_ontology("(subclass HumanIntelligence IntelligenceDiscipline)")


def test_restates_ontology_is_false_for_a_derived_claim():
    assert not restates_ontology("(subclass CITeam CICapability)")


def test_restates_ontology_needs_every_claim_to_be_asserted():
    """A mixed output still carries a claim the source had to supply."""
    mixed = "(subclass HumanIntelligence IntelligenceDiscipline)\n(subclass CITeam CICapability)"
    assert not restates_ontology(mixed)


def test_restates_ontology_is_false_without_subclass_claims():
    assert not restates_ontology("(agent MilitaryProcess AutonomousAgent)")
    assert not restates_ontology("")


def test_ontology_restatements_are_excluded_from_novel_coverage():
    """An accept the ontology already asserts is background knowledge, not formalization."""
    results = evaluate_method(
        [ontology_row()],
        lambda text: "HumanIntelligence subclass-of IntelligenceDiscipline",
        grounder=AcceptingGrounder(),
    )
    metrics = summarize(results)
    assert results[0]["route"] == "accepted"
    assert results[0]["ontology_restated"] is True
    assert metrics["accepted_coverage"] == 1
    assert metrics["accepted_restating_ontology"] == 1
    assert metrics["accepted_novel"] == 0
    assert metrics["accepted_novel_coverage"] == 0


def test_novel_claims_still_count_as_coverage():
    results = evaluate_method(
        [row()], lambda text: "Weapon subclass-of Artifact", grounder=AcceptingGrounder(),
    )
    metrics = summarize(results)
    assert results[0]["ontology_restated"] is False
    assert metrics["accepted_novel"] == 1
    assert metrics["accepted_novel_coverage"] == 1
    assert metrics["accepted_restating_ontology"] == 0


def test_abstention_leaves_restatement_flag_unset():
    results = evaluate_method([row()], lambda text: None)
    assert results[0]["ontology_restated"] is None
    assert summarize(results)["accepted_restating_ontology"] == 0
