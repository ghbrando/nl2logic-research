"""Ask a bounded classification question and retain proof/evidence artifacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.answer import answer_query
from src.reasoning.theory import load_records


def format_answer(result: dict) -> str:
    lines = [f"{result['status'].upper()}: {result['query']}", result["reason"]]
    if result["evidence"]:
        lines.append("\nSupporting premises:")
    for item in result["evidence"]:
        if item["kind"] == "rule":
            lines.append(f"- Rule: {item['description']}")
        elif item["kind"] == "background":
            lines.append(f"- Background assumption: {item['kif']}\n  {item['rationale']}")
        else:
            page = item.get("page_number", item.get("page", "unknown"))
            lines.append(
                f"- Source {item['record_id']}, page {page}: {item['kif']}\n"
                f"  {item.get('original') or item.get('nl', '')}"
            )
    lines.append(f"\nProof and evidence files: {result['artifacts']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True, help="Accepted or reviewed source claims in JSONL")
    parser.add_argument("--background", type=Path, action="append", default=[], help="Explicit background assumptions in JSONL; repeatable")
    parser.add_argument("--query", required=True, help="Ground classification KIF, e.g. (subclass HumanIntelligence Procedure)")
    local = ROOT / "vampire" / "5.0.1" / "vampire.exe"
    parser.add_argument("--vampire", default=os.getenv("VAMPIRE_EXECUTABLE", str(local) if local.exists() else "vampire"))
    parser.add_argument("--timeout", type=int, default=10, help="Seconds per prover invocation (up to three)")
    parser.add_argument("--output-dir", type=Path, default=None, help="New directory for this run's artifacts")
    parser.add_argument("--json", action="store_true", help="Print the full structured result")
    args = parser.parse_args(argv)
    try:
        records = load_records(args.accepted, "source")
        for path in args.background:
            records.extend(load_records(path, "background"))
        result = answer_query(
            records, args.query, executable=args.vampire, timeout=args.timeout,
            output_dir=args.output_dir or ROOT / "results" / "reasoning" / uuid4().hex,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot construct reasoning problem: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else format_answer(result))
    return 0 if result["status"] in {"proven", "disproven"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
