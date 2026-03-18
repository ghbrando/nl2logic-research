#!/usr/bin/env python3
"""Validate FM 2-0 benchmark annotation files.

Checks:
  - Required fields are present in every record
  - record_id is unique across all three files
  - split / status values are from the allowed set
  - Positive examples compile cleanly through CNLCompiler (with --compile)
  - Abstain examples have null pattern / cnl / kif
  - Review examples have formalizable: true

Usage:
    python scripts/validate_benchmark_annotations.py
    python scripts/validate_benchmark_annotations.py --compile
    python scripts/validate_benchmark_annotations.py \\
        --positive path/to/positive.jsonl \\
        --review   path/to/review.jsonl   \\
        --abstain  path/to/abstain.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_POSITIVE = _REPO_ROOT / "data" / "benchmarks" / "fm2-0" / "seed_positive.jsonl"
_DEFAULT_REVIEW   = _REPO_ROOT / "data" / "benchmarks" / "fm2-0" / "seed_review.jsonl"
_DEFAULT_ABSTAIN  = _REPO_ROOT / "data" / "benchmarks" / "fm2-0" / "seed_abstain.jsonl"

_REQUIRED_FIELDS = {
    "record_id", "doc_id", "page", "section", "nl",
    "formalizable", "pattern", "cnl", "kif", "terms",
    "sumo_mapping_confidence", "notes", "status", "split",
}
_VALID_SPLITS    = {"seed", "train", "eval", "abstain"}
_VALID_STATUSES  = {"draft", "reviewed", "accepted"}
_VALID_PATTERNS  = {"instance", "subclass", "relation", "existential", "conditional", "nary"}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [ERROR] {path.name}:{lineno}: JSON parse error: {exc}")
                sys.exit(1)
    return records


def _check_required_fields(record: dict, source: str) -> list[str]:
    errors = []
    for field in sorted(_REQUIRED_FIELDS):
        if field not in record:
            errors.append(f"  [ERROR] {source} record {record.get('record_id', '?')!r}: missing field '{field}'")
    return errors


def _check_enum_fields(record: dict, source: str) -> list[str]:
    errors = []
    rid = record.get("record_id", "?")

    status = record.get("status")
    if status not in _VALID_STATUSES:
        errors.append(f"  [ERROR] {source} {rid!r}: invalid status {status!r} (allowed: {sorted(_VALID_STATUSES)})")

    split = record.get("split")
    if split not in _VALID_SPLITS:
        errors.append(f"  [ERROR] {source} {rid!r}: invalid split {split!r} (allowed: {sorted(_VALID_SPLITS)})")

    pattern = record.get("pattern")
    if pattern is not None and pattern not in _VALID_PATTERNS:
        errors.append(f"  [ERROR] {source} {rid!r}: invalid pattern {pattern!r} (allowed: {sorted(_VALID_PATTERNS)})")

    conf = record.get("sumo_mapping_confidence")
    if conf is not None and conf not in _VALID_CONFIDENCE:
        errors.append(f"  [ERROR] {source} {rid!r}: invalid sumo_mapping_confidence {conf!r} (allowed: {sorted(_VALID_CONFIDENCE)})")

    return errors


def _check_positive(record: dict, source: str) -> list[str]:
    errors = []
    rid = record.get("record_id", "?")
    if not record.get("formalizable"):
        errors.append(f"  [ERROR] {source} {rid!r}: positive record must have formalizable: true")
    if record.get("cnl") is None:
        errors.append(f"  [ERROR] {source} {rid!r}: positive record must have non-null cnl")
    if record.get("kif") is None:
        errors.append(f"  [ERROR] {source} {rid!r}: positive record must have non-null kif")
    if record.get("pattern") is None:
        errors.append(f"  [ERROR] {source} {rid!r}: positive record must have non-null pattern")
    return errors


def _check_abstain(record: dict, source: str) -> list[str]:
    errors = []
    rid = record.get("record_id", "?")
    if record.get("formalizable") is not False:
        errors.append(f"  [ERROR] {source} {rid!r}: abstain record must have formalizable: false")
    for field in ("pattern", "cnl", "kif"):
        if record.get(field) is not None:
            errors.append(f"  [ERROR] {source} {rid!r}: abstain record must have null {field}, got {record[field]!r}")
    return errors


def _compile_positive(records: list[dict]) -> tuple[int, int, list[str]]:
    """Try to compile each positive CNL through CNLCompiler. Returns (ok, fail, errors)."""
    sys.path.insert(0, str(_REPO_ROOT))
    from src.compiler.compiler import CNLCompiler  # noqa: PLC0415
    compiler = CNLCompiler()

    ok = fail = 0
    errors = []
    for record in records:
        cnl = record.get("cnl")
        kif = record.get("kif")
        rid = record.get("record_id", "?")
        if cnl is None:
            continue
        try:
            result = compiler.compile(cnl)
            ok += 1
            if kif and result != kif:
                errors.append(
                    f"  [WARN]  {rid!r}: KIF mismatch\n"
                    f"          compiled : {result!r}\n"
                    f"          annotated: {kif!r}"
                )
        except Exception as exc:  # noqa: BLE001
            fail += 1
            errors.append(f"  [ERROR] {rid!r}: compile failed for CNL {cnl!r}: {exc}")
    return ok, fail, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FM 2-0 benchmark annotation files")
    parser.add_argument("--positive", type=Path, default=_DEFAULT_POSITIVE,
                        help="Path to seed_positive.jsonl")
    parser.add_argument("--review",   type=Path, default=_DEFAULT_REVIEW,
                        help="Path to seed_review.jsonl")
    parser.add_argument("--abstain",  type=Path, default=_DEFAULT_ABSTAIN,
                        help="Path to seed_abstain.jsonl")
    parser.add_argument("--compile",  action="store_true",
                        help="Compile CNL entries in positive file through CNLCompiler")
    args = parser.parse_args()

    all_errors: list[str] = []
    seen_ids: dict[str, str] = {}  # record_id -> source filename

    file_specs = [
        (args.positive, "positive", _check_positive),
        (args.review,   "review",   None),
        (args.abstain,  "abstain",  _check_abstain),
    ]

    file_counts: dict[str, int] = {}

    for path, label, extra_check in file_specs:
        if not path.exists():
            all_errors.append(f"  [ERROR] {label} file not found: {path}")
            file_counts[label] = 0
            continue

        records = _load_jsonl(path)
        file_counts[label] = len(records)

        for record in records:
            source = f"{label}/{path.name}"

            all_errors.extend(_check_required_fields(record, source))
            all_errors.extend(_check_enum_fields(record, source))

            if extra_check:
                all_errors.extend(extra_check(record, source))

            rid = record.get("record_id")
            if rid is not None:
                if rid in seen_ids:
                    all_errors.append(
                        f"  [ERROR] duplicate record_id {rid!r}: "
                        f"appears in both {seen_ids[rid]!r} and {label!r}"
                    )
                else:
                    seen_ids[rid] = label

    # Compile check for positive examples
    compile_ok = compile_fail = 0
    if args.compile and args.positive.exists():
        positive_records = _load_jsonl(args.positive)
        compile_ok, compile_fail, compile_errors = _compile_positive(positive_records)
        all_errors.extend(compile_errors)

    # Summary
    print("-- Benchmark Annotation Validation ----------------------------------")
    print(f"  positive : {file_counts.get('positive', 0)} records")
    print(f"  review   : {file_counts.get('review', 0)} records")
    print(f"  abstain  : {file_counts.get('abstain', 0)} records")
    print(f"  total    : {sum(file_counts.values())} records  ({len(seen_ids)} unique IDs)")
    if args.compile:
        print(f"  compile  : {compile_ok} ok, {compile_fail} failed")
    print()

    validation_errors = [e for e in all_errors if "[ERROR]" in e]
    warnings = [e for e in all_errors if "[WARN]" in e]

    if warnings:
        print("-- Warnings ----------------------------------------------------------")
        for w in warnings:
            print(w)
        print()

    if validation_errors:
        print("-- Errors ------------------------------------------------------------")
        for e in validation_errors:
            print(e)
        print()
        print(f"Validation FAILED: {len(validation_errors)} error(s).")
        sys.exit(1)
    else:
        print("Validation PASSED.")


if __name__ == "__main__":
    main()
