"""Validate source traceability and formal syntax for development claim packets.

This does not decide whether a passage entails a claim; that remains a review task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.compiler import CNLCompiler
from src.ingest.grounding import extract_cnl_terms
from src.ontology.vocab import load_closed_class_terms

_DECLARATION = re.compile(r"\(instance ([A-Z][A-Za-z0-9]*) Class\)\Z")


def _squash(value: str) -> str:
    return " ".join(value.split())


def validate(packet: dict) -> None:
    source = packet["source"]
    pdf = ROOT / source["pdf_path"]
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != source["pdf_sha256"]:
        raise ValueError("PDF hash mismatch")
    pages_path = ROOT / source["pages_path"]
    pages = {r["pdf_page"]: r["text"] for r in map(json.loads, pages_path.read_text(encoding="utf-8").splitlines())}
    candidates_path = ROOT / source["candidates_path"]
    candidates = {r["record_id"]: r for r in map(json.loads, candidates_path.read_text(encoding="utf-8").splitlines())}
    doctrine_axioms = set((ROOT / "data/ontology/doctrine_domain.kif").read_text(encoding="utf-8").splitlines())
    base_classes = load_closed_class_terms()
    seen = set()
    for row in packet["claims"]:
        rid = row["record_id"]
        if rid in seen or rid not in candidates:
            raise ValueError(f"duplicate or unknown record ID: {rid}")
        seen.add(rid)
        candidate = candidates[rid]
        if row["doc_id"] != candidate["doc_id"] or row["pdf_page"] != candidate["pdf_page"] or row["paragraph"] != candidate["paragraph"] or row["page"] != candidate["page"]:
            raise ValueError(f"{rid}: source reference mismatch")
        span = row["source_span"]
        if not span or span not in candidate["source_excerpt"]:
            raise ValueError(f"{rid}: source span absent from paragraph excerpt")
        if _squash(span) not in _squash(pages[row["pdf_page"]]):
            raise ValueError(f"{rid}: source span absent from PDF page extraction")
        if row["coverage"] not in {"full", "partial"}:
            raise ValueError(f"{rid}: invalid coverage")
        if row["status"] != "reviewed" or row["reviewer_type"] not in {"ai", "human"} or not row["reviewed_by"]:
            raise ValueError(f"{rid}: reviewer attribution missing")
        if row["split"] != "development" or not row["claim_text"] or not row["entailment_rationale"] or not isinstance(row["omitted_qualifiers"], list):
            raise ValueError(f"{rid}: incomplete development claim")
        extra_classes = set()
        for declaration in row["declarations"]:
            match = _DECLARATION.fullmatch(declaration)
            if not match or match.group(1) in base_classes:
                raise ValueError(f"{rid}: invalid or redundant class declaration")
            extra_classes.add(match.group(1))
        compiler = CNLCompiler(extra_classes=extra_classes)
        kif = compiler.compile(row["cnl"], require_closed=True, require_argument_kinds=True)
        if kif != row["kif"]:
            raise ValueError(f"{rid}: compiled KIF mismatch")
        if len(kif.splitlines()) != 1 or kif in doctrine_axioms:
            raise ValueError(f"{rid}: compound claim or preloaded ontology assertion")
        _, _, terms = extract_cnl_terms(row["cnl"])
        if set(terms) != set(row["terms"]):
            raise ValueError(f"{rid}: term list mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        validate(packet)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"Claim packet invalid: {exc}\n")
    print(f"Validated {len(packet['claims'])} development claim(s); entailment still requires review")


if __name__ == "__main__":
    main()
