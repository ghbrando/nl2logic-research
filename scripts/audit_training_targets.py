"""Check every NL→CNL training target against the closed compiler."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.compiler import CNLCompiler


def error_kind(exc: Exception) -> str:
    message = str(exc)
    if "Unbound variables:" in message:
        return "unbound_variable"
    if "No terminal matches" in message or "Unexpected" in message:
        return "grammar_or_token"
    if "Unknown" in message or "not in" in message:
        return "vocabulary_or_type"
    return type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = CNLCompiler()
    counts = Counter()
    by_pattern: dict[str, Counter] = {}
    examples: dict[str, dict] = {}
    with args.train_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pattern = row.get("pattern") or "unknown"
            pattern_counts = by_pattern.setdefault(pattern, Counter())
            counts["total"] += 1
            pattern_counts["total"] += 1
            try:
                compiled = compiler.compile(row["cnl"], require_closed=True)
            except Exception as exc:
                kind = error_kind(exc)
                counts[kind] += 1
                pattern_counts[kind] += 1
                examples.setdefault(kind, {"line": line_number, "pattern": pattern,
                                           "cnl": row["cnl"], "error": str(exc)[:300]})
                continue
            counts["valid_closed_cnl"] += 1
            pattern_counts["valid_closed_cnl"] += 1
            if row.get("kif") != compiled:
                counts["kif_mismatch"] += 1
                pattern_counts["kif_mismatch"] += 1
                examples.setdefault("kif_mismatch", {"line": line_number,
                                 "pattern": pattern, "cnl": row["cnl"],
                                 "expected_kif": row.get("kif"), "compiled_kif": compiled})
    report = {
        "train_file": str(args.train_file),
        "sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "counts": dict(counts),
        "by_pattern": {key: dict(value) for key, value in sorted(by_pattern.items())},
        "examples": examples,
        "interpretation": "A syntactic/scope/KIF consistency audit, not a semantic judgment of NL or doctrine.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
