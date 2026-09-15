"""Audit how much of a doctrine train/eval split the ontology already answers.

The doctrine KIF is not inert background. It supplies implied parents to the
prompt router that writes a subclass claim and to the grounding gate that
licenses that claim's direction, so a target it already asserts can be produced
and validated without the source sentence carrying it.

A train/eval split can therefore pass an exact-text overlap check and still be
uninformative: if both halves are drawn from the same ontology file, the split
measures recall of a closed vocabulary the system already has, not
formalization of a passage.

This reports, per file, how many targets the ontology already asserts, and
whether the training and benchmark targets partition one shared pool.

Run it before quoting any coverage number:

  python scripts/audit_ontology_leakage.py \
      --training data/training_pairs/doctrine_real_train.jsonl \
      --benchmark data/benchmarks/fm2-0/seed_positive.jsonl \
      --benchmark data/benchmarks/fm2-0/seed_review.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.comparison import load_asserted_subclass_pairs, subclass_claims

_DEFAULT_ONTOLOGY = _REPO_ROOT / "data" / "ontology" / "doctrine_domain.kif"


def read_targets(path: Path) -> list[tuple[str, str]]:
    """Every subclass claim appearing as a gold target in a JSONL file.

    Records without a `kif` field are abstention examples and contribute no
    target; they are counted separately by the caller.
    """
    claims: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        claims.extend(subclass_claims(json.loads(line).get("kif") or ""))
    return claims


def audit(ontology: Path, training: list[Path], benchmark: list[Path]) -> dict:
    asserted = load_asserted_subclass_pairs(ontology)

    def summarize(paths: list[Path]) -> tuple[set[tuple[str, str]], list[dict]]:
        pool: set[tuple[str, str]] = set()
        per_file = []
        for path in paths:
            claims = read_targets(path)
            restated = [c for c in claims if c in asserted]
            pool.update(claims)
            per_file.append({
                "file": str(path).replace("\\", "/"),
                "targets": len(claims),
                "restating_ontology": len(restated),
                "novel": len(claims) - len(restated),
            })
        return pool, per_file

    train_pool, train_files = summarize(training)
    bench_pool, bench_files = summarize(benchmark)

    train_in = train_pool & asserted
    bench_in = bench_pool & asserted
    return {
        "ontology": str(ontology).replace("\\", "/"),
        "ontology_subclass_assertions": len(asserted),
        "training_files": train_files,
        "benchmark_files": bench_files,
        "ontology_used_as_training_target": len(train_in),
        "ontology_used_as_benchmark_target": len(bench_in),
        "ontology_used_by_both": len(train_in & bench_in),
        "ontology_never_used": len(asserted - train_pool - bench_pool),
        "training_targets_outside_ontology": sorted(
            " ".join(c) for c in (train_pool - asserted)
        ),
        "benchmark_targets_outside_ontology": sorted(
            " ".join(c) for c in (bench_pool - asserted)
        ),
        "split_partitions_one_ontology_pool": bool(
            train_in and bench_in and not (train_in & bench_in)
            and not (train_pool - asserted) and len(bench_pool - asserted) <= 1
        ),
    }


def render(report: dict) -> str:
    lines = [
        f"Ontology: {report['ontology']}",
        f"  subclass assertions            : {report['ontology_subclass_assertions']}",
        f"  used as training targets       : {report['ontology_used_as_training_target']}",
        f"  used as benchmark targets      : {report['ontology_used_as_benchmark_target']}",
        f"  used by both                   : {report['ontology_used_by_both']}",
        f"  never used                     : {report['ontology_never_used']}",
        "",
        f"{'file':<58} {'targets':>8} {'restated':>9} {'novel':>6}",
    ]
    for group in ("training_files", "benchmark_files"):
        for row in report[group]:
            lines.append(
                f"{row['file']:<58} {row['targets']:>8} {row['restating_ontology']:>9} {row['novel']:>6}"
            )
    novel_bench = sum(r["novel"] for r in report["benchmark_files"])
    lines.append("")
    if report["split_partitions_one_ontology_pool"]:
        lines.append(
            "VERDICT: the training and benchmark targets partition one ontology pool."
        )
        lines.append(
            "  They do not overlap each other, so an exact-text leakage check passes,"
        )
        lines.append(
            "  but both are drawn from the file that also licenses acceptance. Coverage"
        )
        lines.append(
            "  measured on this split reflects a closed vocabulary, not formalization."
        )
    elif novel_bench == 0:
        lines.append(
            "VERDICT: every benchmark target is already asserted in the ontology."
        )
    else:
        lines.append(
            f"VERDICT: {novel_bench} benchmark target(s) are not asserted in the ontology."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit ontology leakage in a doctrine split")
    parser.add_argument("--ontology", type=Path, default=_DEFAULT_ONTOLOGY)
    parser.add_argument("--training", type=Path, action="append", default=[])
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if not args.training and not args.benchmark:
        parser.error("supply at least one --training or --benchmark file")
    for path in [args.ontology, *args.training, *args.benchmark]:
        if not path.exists():
            parser.error(f"missing file: {path}")

    report = audit(args.ontology, args.training, args.benchmark)
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
