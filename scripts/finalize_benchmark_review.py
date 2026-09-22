"""Validate explicitly reviewed CSV rows and export evaluation gold JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.compiler import CNLCompiler
from src.ingest.grounding import extract_cnl_terms


def finalize(candidates: list[dict], annotations: list[dict], compiler=None) -> list[dict]:
    compiler = compiler or CNLCompiler()
    by_id = {r["record_id"]: r for r in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    gold, seen = [], set()
    for annotation in annotations:
        rid = annotation.get("record_id")
        if rid not in by_id or rid in seen:
            raise ValueError(f"Unknown or duplicate annotation ID: {rid}")
        seen.add(rid)
        if annotation.get("status") != "reviewed":
            continue
        source = by_id[rid]
        if annotation.get("nl") != source["nl"]:
            raise ValueError(f"{rid}: source text changed; revise/version the candidate before annotation")
        if not (annotation.get("reviewed_by") or "").strip():
            raise ValueError(f"{rid}: reviewed_by is required")
        reviewer_type = (annotation.get("reviewer_type") or "unspecified").strip().lower()
        if reviewer_type not in {"human", "ai", "unspecified"}:
            raise ValueError(f"{rid}: reviewer_type must be human, ai, or unspecified")
        label = (annotation.get("formalizable") or "").strip().lower()
        if label not in {"true", "false"}:
            raise ValueError(f"{rid}: formalizable must be true or false")
        cnl = (annotation.get("cnl") or "").strip() or None
        pattern = (annotation.get("pattern") or "").strip() or None
        if label == "true":
            if not cnl or pattern not in {"subclass", "instance", "relation", "binary", "nary", "existential", "conditional", "negation"}:
                raise ValueError(f"{rid}: positive annotation needs CNL and a supported pattern")
            try:
                kif = compiler.compile(cnl)
            except Exception as exc:
                raise ValueError(f"{rid}: invalid reviewed CNL: {exc}") from exc
        else:
            if cnl or pattern:
                raise ValueError(f"{rid}: abstention annotation must leave CNL/pattern empty")
            kif = None
        seconds = annotation.get("review_seconds")
        seconds = float(seconds) if seconds else None
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"{rid}: invalid review duration")
        gold.append({**source, "formalizable": label == "true", "cnl": cnl, "kif": kif,
                     "terms": extract_cnl_terms(cnl)[2] if cnl else [],
                     "pattern": pattern, "status": "reviewed", "split": "eval",
                     "reviewed_by": annotation["reviewed_by"].strip(),
                     "reviewer_type": reviewer_type, "review_seconds": seconds,
                     "notes": annotation.get("notes", "")})
    return gold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        candidates = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
        with args.annotations.open(newline="", encoding="utf-8-sig") as handle:
            gold = finalize(candidates, list(csv.DictReader(handle)))
        if not gold:
            raise ValueError("No explicitly reviewed rows; no gold artifact written")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Review validation failed: {exc}\n")
    print(f"Exported {len(gold)} reviewed records to {args.output}")


if __name__ == "__main__":
    main()
