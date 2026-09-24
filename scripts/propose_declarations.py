"""Propose isolated class declarations from unlabeled sources alone.

Each extracted copular definition head whose defined phrase is not already an
ontology class receives a sense-specific symbol. The registry asserts no
parent and reads no labels, predictions, or reviewed packets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import text_sha256
from scripts.run_partial_claim_dev_probe import load_sources
from src.ontology.vocab import load_closed_class_terms
from src.preprocessing.declarations import (
    PROPOSED_STATUS, load_declaration_registry, propose_declaration,
)
from src.preprocessing.partial_claims import extract_paragraph_candidate

PARENT_POOL = ["Process", "IntelligenceProduct"]


def propose(source_path: Path, source_manifest_path: Path, output_path: Path) -> dict:
    rows = load_sources(source_path, source_manifest_path)
    known = load_closed_class_terms()
    declarations, skipped = [], []
    for row in rows:
        candidate, screened = extract_paragraph_candidate(row["source_excerpt"], row["source_sentence"])
        if candidate.status != "candidate" or candidate.gate_reasons:
            continue
        evidence = row["source_excerpt"][screened[-1]["start"]:screened[-1]["end"]]
        entry = propose_declaration(candidate.candidate_text, known, evidence)
        if entry is None:
            skipped.append({"record_id": row["record_id"], "candidate_text": candidate.candidate_text,
                            "reason": "declined_existing_class_or_non_kind_genus"})
        elif any(existing["symbol"] == entry["symbol"] or existing["alias"].casefold() == entry["alias"].casefold()
                 for existing in declarations):
            skipped.append({"record_id": row["record_id"], "candidate_text": candidate.candidate_text,
                            "reason": "duplicate_alias_or_symbol"})
        else:
            declarations.append(entry)
    registry = {
        "status": PROPOSED_STATUS,
        "sources_sha256": text_sha256(source_path),
        "generator": "scripts/propose_declarations.py",
        "existing_parent_pool": PARENT_POOL,
        "declarations": declarations,
        "skipped": skipped,
        "limitation": "Source-only proposal: symbols join the defined phrase and the head of its article-introduced genus noun phrase. They are unreviewed and assert no parent.",
    }
    output_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_declaration_registry(output_path, sources_sha256=registry["sources_sha256"])
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = propose(args.sources, args.source_manifest, args.output)
    print(json.dumps({"declarations": len(registry["declarations"]), "skipped": len(registry["skipped"])}))


if __name__ == "__main__":
    main()
