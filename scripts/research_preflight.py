"""Check data prerequisites for a source-only research comparison (no GPU needed).

This is a necessary data gate, not certification of a blind or adequate study.
Exit 2 means prerequisites are missing; the JSON report explains each blocker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.compiler import CNLCompiler
from src.eval.comparison import fingerprint, read_benchmark, restates_ontology, training_overlap


def assess(paths: list[Path], training_paths: list[Path]) -> dict:
    records = read_benchmark(paths)
    overlap = training_overlap(records, training_paths)
    reviewed = [r for r in records if r["status"] in {"reviewed", "accepted"}
                and r.get("split") != "train"]
    eligible = [r for r in reviewed if r["record_id"] not in overlap]
    positive = [r for r in eligible if r["formalizable"]]
    novel = [r for r in positive if not restates_ontology(r["kif"])]
    scope_errors = {}
    compiler = CNLCompiler()
    for row in positive:
        try:
            compiler.compile(row["cnl"], require_closed=True)
        except ValueError as exc:
            scope_errors[row["record_id"]] = str(exc)
    blockers = []
    if not training_paths:
        blockers.append("Actual training files must be supplied for overlap checking.")
    if overlap:
        blockers.append("Resolve training/benchmark exact-text overlap before freezing the study.")
    if not novel:
        blockers.append("No eligible reviewed positive targets beyond direct ontology restatements.")
    if not any(not r["formalizable"] for r in eligible):
        blockers.append("No eligible reviewed abstention targets.")
    if scope_errors:
        blockers.append("Gold formulas contain unbound variables.")
    return {
        "status": "blocked" if blockers else "data_checks_passed",
        "background_policy": "source-only", "blockers": blockers,
        "counts": {"records": len(records), "reviewed_test": len(reviewed),
                   "eligible": len(eligible), "positive": len(positive),
                   "beyond_direct_restatements": len(novel)},
        "training_text_overlap": overlap, "scope_errors": scope_errors,
        "benchmark_files": [fingerprint(p) for p in paths],
        "training_files": [fingerprint(p) for p in training_paths],
        "limitations": ["Novelty check covers direct subclass restatements only, not logical entailment.",
                        "Exact-text checks do not detect paraphrase or document leakage.",
                        "Human label quality, sample size and blind split independence need separate review.",
                        "No model execution or hardware validation is performed."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--training-data", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = assess(args.benchmark, args.training_data)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot check research data: {exc}\n")
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
