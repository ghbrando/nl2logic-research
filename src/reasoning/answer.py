"""Consistency-first answers for the classification fragment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .theory import UnsupportedFormula, build_theory, parse_literal
from .vampire import ProverResult, proof_inputs, run_vampire


def answer_query(
    records: list[dict], query: str, *, executable: str,
    output_dir: Path, timeout: int = 10,
    runner: Callable = run_vampire,
) -> dict:
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    # Refuse to overwrite an earlier run's evidence.
    output_dir.mkdir(parents=True, exist_ok=False)
    theory, evidence, terms = build_theory(records)
    (output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "theory.p").write_text(theory, encoding="utf-8")
    runs = {}

    def finish(status: str, reason: str, names: list[str] | None = None) -> dict:
        result = {
            "status": status, "reason": reason, "query": query,
            "theory_sha256": hashlib.sha256(theory.encode("utf-8")).hexdigest(),
            "evidence": [{"axiom_name": name, **evidence[name]} for name in names or []],
            "runs": runs, "artifacts": str(output_dir.resolve()),
            "semantics": "Ground classification fragment; open world; explicit negation; two inheritance rules. Not full SUMO.",
        }
        (output_dir / "answer.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    try:
        literal = parse_literal(query)
    except UnsupportedFormula as exc:
        return finish("unknown", f"Unsupported question: {exc}")
    if literal.left not in terms or literal.right not in terms:
        return finish("unknown", "Question contains terms absent from the loaded theory")

    def run(name: str, conjecture=None) -> ProverResult:
        problem = output_dir / f"{name}.p"
        content = theory
        if conjecture is not None:
            content += f"fof(query,conjecture,({conjecture.tptp()})).\n"
        problem.write_text(content, encoding="utf-8")
        outcome = runner(executable, problem, timeout)
        (output_dir / f"{name}.stdout.txt").write_text(outcome.stdout, encoding="utf-8")
        (output_dir / f"{name}.stderr.txt").write_text(outcome.stderr, encoding="utf-8")
        runs[name] = {
            "status": outcome.status, "returncode": outcome.returncode,
            "executable": executable, "timeout_seconds": timeout,
            "problem": str(problem.resolve()),
        }
        return outcome

    consistency = run("consistency")
    if consistency.status == "Unsatisfiable":
        names = proof_inputs(consistency.stdout, evidence)
        if names is None:
            return finish("unknown", "Inconsistency reported without traceable proof")
        return finish("inconsistent", "Loaded premises contradict one another", names)
    if consistency.status != "Satisfiable":
        return finish("unknown", f"Consistency not established: {consistency.status}")

    for name, goal, answer in (("positive", literal, "proven"), ("negative", literal.negate(), "disproven")):
        outcome = run(name, goal)
        if outcome.status == "Theorem":
            names = proof_inputs(outcome.stdout, evidence)
            if names is None:
                return finish("unknown", "Theorem reported without traceable proof")
            return finish(answer, "Query follows from the theory" if answer == "proven" else "Explicit negation of the query follows from the theory", names)
        if outcome.status != "CounterSatisfiable":
            return finish("unknown", f"Proof search incomplete: {outcome.status}")
    return finish("unknown", "Neither the query nor its negation follows from the loaded theory")
