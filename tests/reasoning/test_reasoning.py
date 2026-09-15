from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.reasoning.answer import answer_query
from src.reasoning.theory import UnsupportedFormula, build_theory, load_records, parse_literal
from src.reasoning.vampire import ProverResult, proof_inputs, run_vampire

ROOT = Path(__file__).resolve().parents[2]


def record(kif, name="claim"):
    return {"record_id": name, "kif": kif, "kind": "source", "original": "Synthetic test premise", "page": 1}


@pytest.mark.parametrize("kif", [
    "(subclass A)", "(subclass A B C)", "(agent A B)",
    "(subclass ?x B)", "(forall (?x) (instance ?x B))",
    "(subclass A B) (subclass C D)", "(subclass A 'b')",
    "(subclass A B). fof(injected,axiom,$false).", "(instance (Fn A) B)",
])
def test_unsupported_kif_fails_closed(kif):
    with pytest.raises(UnsupportedFormula):
        parse_literal(kif)


def test_translation_preserves_polarity_predicate_and_case():
    literal = parse_literal(" (not (instance ExampleUnit MilitaryUnit)) ")
    assert literal.tptp() == "~(instance('ExampleUnit','MilitaryUnit'))"
    assert literal.negate().tptp() == "instance('ExampleUnit','MilitaryUnit')"


def test_loader_does_not_promote_draft_annotations(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({**record("(subclass A B)"), "status": "draft"}))
    with pytest.raises(ValueError, match="draft"):
        load_records(path, "source")


def test_background_requires_explicit_rationale(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record("(subclass A B)")))
    with pytest.raises(ValueError, match="rationale"):
        load_records(path, "background")


@pytest.mark.parametrize("failure", ["Timeout", "Error", "GaveUp", "Unknown"])
def test_incomplete_consistency_check_never_answers(tmp_path, failure):
    calls = []
    def runner(*args):
        calls.append(args)
        return ProverResult(failure)
    result = answer_query([record("(subclass A B)")], "(subclass A B)",
                          executable="unused", output_dir=tmp_path / "run", runner=runner)
    assert result["status"] == "unknown"
    assert result["evidence"] == []
    assert len(calls) == 1


def test_theorem_without_proof_is_unknown(tmp_path):
    outcomes = iter([ProverResult("Satisfiable"), ProverResult("Theorem")])
    result = answer_query([record("(subclass A B)")], "(subclass A B)",
                          executable="unused", output_dir=tmp_path / "run", runner=lambda *a: next(outcomes))
    assert result["status"] == "unknown"


def test_timeout_during_query_never_becomes_disproven(tmp_path):
    outcomes = iter([ProverResult("Satisfiable"), ProverResult("Timeout")])
    result = answer_query([record("(subclass A B)")], "(subclass A B)",
                          executable="unused", output_dir=tmp_path / "run", runner=lambda *a: next(outcomes))
    assert result["status"] == "unknown"
    assert "Timeout" in result["reason"]


def test_external_timeout_is_bounded(monkeypatch, tmp_path):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 5
        assert "--output_axiom_names" in command
        raise subprocess.TimeoutExpired(command, 5)
    monkeypatch.setattr(subprocess, "run", timeout)
    assert run_vampire("vampire", tmp_path / "p.p", 3).status == "Timeout"


def test_failed_process_cannot_claim_a_theorem(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "% SZS status Theorem\n", "crash"))
    assert run_vampire("vampire", tmp_path / "p.p").status == "Error"


def test_missing_executable_returns_error(tmp_path):
    assert run_vampire(str(tmp_path / "absent-executable"), tmp_path / "p.p").status == "Error"


def test_unsupported_question_does_not_invoke_prover(tmp_path):
    def unexpected(*args):
        pytest.fail("Unsupported query must not reach prover")
    result = answer_query([record("(subclass A B)")], "(agent A B)",
                          executable="unused", output_dir=tmp_path / "run", runner=unexpected)
    assert result["status"] == "unknown"


def test_proof_evidence_does_not_use_echoed_input():
    assert proof_inputs("file('p',axiom_0). $false", {"axiom_0": {}}) is None


@pytest.fixture
def vampire():
    local = ROOT / "vampire" / "5.0.1" / "vampire.exe"
    executable = os.getenv("VAMPIRE_EXECUTABLE") or (str(local) if local.exists() else shutil.which("vampire"))
    if not executable:
        pytest.skip("Install Vampire or set VAMPIRE_EXECUTABLE for actual prover tests")
    return executable


@pytest.mark.parametrize("facts,query,status", [
    (["(subclass A B)", "(subclass B C)"], "(subclass A C)", "proven"),
    (["(instance Unit A)", "(subclass A B)"], "(instance Unit B)", "proven"),
    (["(not (subclass A B))"], "(subclass A B)", "disproven"),
    (["(subclass A B)", "(subclass C D)"], "(subclass A D)", "unknown"),
    (["(subclass A B)", "(subclass B C)", "(not (subclass A C))"], "(subclass A B)", "inconsistent"),
    (["(subclass A B)", "(not (subclass A B))", "(subclass C D)"], "(subclass C D)", "inconsistent"),
    (["(subclass A B)"], "(not (subclass A B))", "disproven"),
])
def test_actual_vampire_outcomes(vampire, tmp_path, facts, query, status):
    records = [record(kif, f"r{i}") for i, kif in enumerate(facts)]
    result = answer_query(records, query, executable=vampire, output_dir=tmp_path / "run")
    assert result["status"] == status, result
    if status != "unknown":
        assert result["evidence"]
    if status == "inconsistent":
        assert set(result["runs"]) == {"consistency"}
    assert json.loads((tmp_path / "run" / "answer.json").read_text())["status"] == status


def test_actual_doctrine_proof_traces_source_and_background(vampire, tmp_path):
    records = load_records(ROOT / "data/benchmarks/fm2-0/seed_positive.jsonl", "source")
    records += load_records(ROOT / "examples/reasoning/background.jsonl", "background")
    result = answer_query(records, "(subclass HumanIntelligence Procedure)",
                          executable=vampire, output_dir=tmp_path / "run")
    assert result["status"] == "proven", result
    assert {row["kind"] for row in result["evidence"]} == {"source", "background", "rule"}
    sources = [row for row in result["evidence"] if row["kind"] == "source"]
    assert [row["record_id"] for row in sources] == ["fm2-0-016"]
    assert sources[0]["page"] == 18
