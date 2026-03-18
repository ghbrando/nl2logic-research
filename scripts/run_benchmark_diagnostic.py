#!/usr/bin/env python3
"""Phase 0 FM 2-0 benchmark diagnostic runner.

Runs the constrained NL->CNL->KIF pipeline against the seed benchmark and
buckets each example into one of seven outcome categories:

  correct           - pipeline output matches gold CNL and KIF (positive only)
  wrong_kif         - CNL matches gold but KIF differs (compile divergence)
  wrong_cnl         - pipeline produced a CNL but it does not match gold
  grounding_rejected - pipeline produced a CNL but DoctrineGrounder rejected it
  compile_failed    - pipeline produced a CNL string that fails to compile
  abstained         - CNLSampler abstained (correct for abstain set; wrong for positive)
  review_only       - record is in seed_review; counted separately, not scored

Usage:
    # Gold-passthrough mode (no model needed -- score gold against itself):
    python scripts/run_benchmark_diagnostic.py --gold-passthrough

    # With a fine-tuned model:
    python scripts/run_benchmark_diagnostic.py --model-path path/to/model

    # Custom output location:
    python scripts/run_benchmark_diagnostic.py --gold-passthrough --output results/diag.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BENCHMARK_DIR = _REPO_ROOT / "data" / "benchmarks" / "fm2-0"
_DEFAULT_POSITIVE = _BENCHMARK_DIR / "seed_positive.jsonl"
_DEFAULT_REVIEW   = _BENCHMARK_DIR / "seed_review.jsonl"
_DEFAULT_ABSTAIN  = _BENCHMARK_DIR / "seed_abstain.jsonl"
_DEFAULT_OUTPUT   = _REPO_ROOT / "results" / "diagnostic_latest.jsonl"

# Bucket names (ordered for summary display)
_BUCKETS = [
    "correct",
    "wrong_kif",
    "wrong_cnl",
    "grounding_rejected",
    "compile_failed",
    "abstained",
    "review_only",
    "error",
]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    record_id: str
    doc_id: str
    page: int | None
    split: str
    nl: str
    expected_formalizable: bool
    gold_cnl: str | None
    gold_kif: str | None
    predicted_cnl: str | None
    predicted_kif: str | None
    compiled_kif: str | None
    grounding_accepted: bool | None
    grounding_reason: str | None
    bucket: str
    notes: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [ERROR] {path.name}:{lineno}: JSON parse error: {exc}", file=sys.stderr)
                sys.exit(1)
    return records


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    nl: str,
    sampler,
    compiler,
    grounder,
) -> tuple[str | None, str | None, bool | None, str | None, str | None]:
    """Return (predicted_cnl, compiled_kif, grounding_accepted, grounding_reason, error)."""
    from src.fsm.cnl_fsm import UnsupportedInputError

    # Step 1: abstention check
    if sampler is None:
        return None, None, None, None, "no_model"

    try:
        from src.training.train import format_prompt
        prompt = format_prompt(nl)
        predicted_cnl = sampler.sample(prompt)
    except UnsupportedInputError:
        return None, None, None, None, None  # abstained -- not an error
    except Exception as exc:
        return None, None, None, None, f"sampler: {exc}"

    # Step 2: compile
    try:
        compiled_kif = compiler.compile(predicted_cnl)
    except Exception as exc:
        return predicted_cnl, None, None, None, f"compile: {exc}"

    # Step 3: grounding
    try:
        assessment = grounder.assess(
            cnl=predicted_cnl,
            original_text=nl,
            normalized_text=nl,
            linked_text=nl,
            subclaim_text=nl,
        )
        return predicted_cnl, compiled_kif, assessment.accepted, assessment.reason, None
    except Exception as exc:
        return predicted_cnl, compiled_kif, None, None, f"grounding: {exc}"


def _gold_passthrough(
    nl: str,
    gold_cnl: str | None,
    gold_kif: str | None,
    compiler,
    grounder,
) -> tuple[str | None, str | None, bool | None, str | None, str | None]:
    """Treat gold CNL as the prediction; compile and ground it."""
    if gold_cnl is None:
        return None, None, None, None, None  # abstained for review/abstain records

    try:
        compiled_kif = compiler.compile(gold_cnl)
    except Exception as exc:
        return gold_cnl, None, None, None, f"compile: {exc}"

    try:
        assessment = grounder.assess(
            cnl=gold_cnl,
            original_text=nl,
            normalized_text=nl,
            linked_text=nl,
            subclaim_text=nl,
        )
        return gold_cnl, compiled_kif, assessment.accepted, assessment.reason, None
    except Exception as exc:
        return gold_cnl, compiled_kif, None, None, f"grounding: {exc}"


# ---------------------------------------------------------------------------
# Bucketing logic
# ---------------------------------------------------------------------------

def _bucket(
    *,
    record: dict[str, Any],
    predicted_cnl: str | None,
    compiled_kif: str | None,
    grounding_accepted: bool | None,
    grounding_reason: str | None,
    error: str | None,
) -> str:
    gold_cnl   = record.get("cnl")
    gold_kif   = record.get("kif")
    split      = record.get("split", "seed")
    formalizable = record.get("formalizable", True)

    # Review records are never scored
    if split == "abstain" or record.get("status") == "reviewed" and not formalizable:
        # abstain set: correct outcome is abstention
        if predicted_cnl is None and error is None:
            return "correct"
        if predicted_cnl is None and error == "no_model":
            return "abstained"  # can't evaluate without model
        return "wrong_cnl"  # pipeline produced something it shouldn't

    # Records from seed_review file
    if record.get("_source") == "review":
        return "review_only"

    # Pipeline error (not abstention)
    if error and not error.startswith("compile:"):
        if error == "no_model":
            return "abstained"
        return "error"

    # Abstained
    if predicted_cnl is None:
        if not formalizable:
            return "correct"
        return "abstained"

    # Compile failed
    if compiled_kif is None:
        return "compile_failed"

    # Grounding rejected
    if grounding_accepted is False:
        return "grounding_rejected"

    # CNL matches gold?
    cnl_match = (predicted_cnl.strip() == (gold_cnl or "").strip())
    kif_match = (compiled_kif.strip() == (gold_kif or "").strip())

    if cnl_match and kif_match:
        return "correct"
    if cnl_match and not kif_match:
        return "wrong_kif"
    return "wrong_cnl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 FM 2-0 benchmark diagnostic runner"
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to fine-tuned model directory. If omitted, --gold-passthrough is assumed."
    )
    parser.add_argument(
        "--gold-passthrough", action="store_true",
        help="Score gold annotations against the compiler/grounder without a model."
    )
    parser.add_argument(
        "--positive", type=Path, default=_DEFAULT_POSITIVE,
        help="Path to seed_positive.jsonl"
    )
    parser.add_argument(
        "--review", type=Path, default=_DEFAULT_REVIEW,
        help="Path to seed_review.jsonl"
    )
    parser.add_argument(
        "--abstain", type=Path, default=_DEFAULT_ABSTAIN,
        help="Path to seed_abstain.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Output JSONL path for per-record results"
    )
    args = parser.parse_args()

    if not args.gold_passthrough and args.model_path is None:
        print(
            "No --model-path supplied. Defaulting to --gold-passthrough mode.\n"
            "Supply --model-path to run the full constrained pipeline.",
            file=sys.stderr,
        )
        args.gold_passthrough = True

    # Load infrastructure
    from src.compiler.compiler import CNLCompiler
    from src.ingest.grounding import DoctrineGrounder

    compiler = CNLCompiler()
    grounder = DoctrineGrounder()

    sampler = None
    if args.model_path is not None:
        try:
            from src.eval.model_predictor import load_model_and_tokenizer
            from src.fsm.cnl_fsm import CNLSampler
            model, tokenizer = load_model_and_tokenizer(args.model_path)
            sampler = CNLSampler(model, tokenizer)
            print(f"Model loaded from {args.model_path}")
        except Exception as exc:
            print(f"[ERROR] Could not load model: {exc}", file=sys.stderr)
            sys.exit(1)

    # Load benchmark records
    file_specs = [
        (args.positive, "positive"),
        (args.review,   "review"),
        (args.abstain,  "abstain"),
    ]
    all_records: list[dict[str, Any]] = []
    for path, label in file_specs:
        if not path.exists():
            print(f"  [WARN] {label} file not found: {path}", file=sys.stderr)
            continue
        for rec in _load_jsonl(path):
            rec["_source"] = label
            all_records.append(rec)

    print(f"Loaded {len(all_records)} records from benchmark.")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results: list[DiagnosticResult] = []
    bucket_counts: dict[str, int] = {b: 0 for b in _BUCKETS}

    for rec in all_records:
        record_id = rec.get("record_id", "?")
        nl        = rec.get("nl", "")
        gold_cnl  = rec.get("cnl")
        gold_kif  = rec.get("kif")

        try:
            if args.gold_passthrough:
                predicted_cnl, compiled_kif, grounding_accepted, grounding_reason, error = (
                    _gold_passthrough(nl, gold_cnl, gold_kif, compiler, grounder)
                )
            else:
                predicted_cnl, compiled_kif, grounding_accepted, grounding_reason, error = (
                    _run_pipeline(nl, sampler, compiler, grounder)
                )
        except Exception as exc:
            predicted_cnl  = None
            compiled_kif   = None
            grounding_accepted = None
            grounding_reason   = None
            error = f"unexpected: {exc}"

        bucket = _bucket(
            record=rec,
            predicted_cnl=predicted_cnl,
            compiled_kif=compiled_kif,
            grounding_accepted=grounding_accepted,
            grounding_reason=grounding_reason,
            error=error,
        )

        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        notes_parts = []
        if error:
            notes_parts.append(f"error={error}")
        if grounding_reason:
            notes_parts.append(f"grounding_reason={grounding_reason}")

        result = DiagnosticResult(
            record_id=record_id,
            doc_id=rec.get("doc_id", ""),
            page=rec.get("page"),
            split=rec.get("split", ""),
            nl=nl,
            expected_formalizable=bool(rec.get("formalizable", True)),
            gold_cnl=gold_cnl,
            gold_kif=gold_kif,
            predicted_cnl=predicted_cnl,
            predicted_kif=gold_kif if args.gold_passthrough else None,
            compiled_kif=compiled_kif,
            grounding_accepted=grounding_accepted,
            grounding_reason=grounding_reason,
            bucket=bucket,
            notes="; ".join(notes_parts),
            error=error,
        )
        results.append(result)

    # Write JSONL output
    with open(args.output, "w", encoding="utf-8") as fout:
        for result in results:
            fout.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    # Console summary
    total = len(results)
    review_count   = bucket_counts.get("review_only", 0)
    scoreable      = total - review_count

    positive_correct = sum(
        1 for r in results
        if r.bucket == "correct" and r.expected_formalizable
    )
    abstain_correct = sum(
        1 for r in results
        if r.bucket == "correct" and not r.expected_formalizable
    )

    mode_label = "gold-passthrough" if args.gold_passthrough else f"model={args.model_path}"

    print()
    print("-- FM 2-0 Seed Benchmark Diagnostic ----------------------------------")
    print(f"  mode        : {mode_label}")
    print(f"  total       : {total} records  ({scoreable} scored, {review_count} review-only)")
    print()
    print("  Bucket breakdown:")
    for bucket in _BUCKETS:
        count = bucket_counts.get(bucket, 0)
        pct   = f"{100 * count / total:.0f}%" if total else "-"
        print(f"    {bucket:<22} {count:>3}  ({pct})")
    print()
    if scoreable > 0:
        pct_correct = 100 * (positive_correct + abstain_correct) / scoreable
        print(f"  Correct (positive) : {positive_correct}")
        print(f"  Correct (abstain)  : {abstain_correct}")
        print(f"  Overall accuracy   : {pct_correct:.1f}%  ({positive_correct + abstain_correct}/{scoreable} scored)")
    print()
    print(f"  Results written to : {args.output}")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
