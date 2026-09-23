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
_ABSTENTION_REASONS = {
    "preloaded_axiom", "scope_or_qualifier", "cardinality", "disjunction",
    "relation_semantics", "comparison", "representation_gap", "causality",
    "temporal_context", "named_entity", "incomplete_source", "nonassertive_reference",
    "modality_or_quantifier",
}


def _squash(value: str) -> str:
    return " ".join(value.split())


def _text_sha256(path: Path) -> str:
    """Hash the Git-canonical LF form on Windows and Linux alike."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate(packet: dict) -> None:
    source = packet["source"]
    pdf = ROOT / source["pdf_path"]
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != source["pdf_sha256"]:
        raise ValueError("PDF hash mismatch")
    pages_path = ROOT / source["pages_path"]
    pages = {r["pdf_page"]: r["text"] for r in map(json.loads, pages_path.read_text(encoding="utf-8").splitlines())}
    candidates_path = ROOT / source["candidates_path"]
    candidates = {r["record_id"]: r for r in map(json.loads, candidates_path.read_text(encoding="utf-8").splitlines())}
    split = packet.get("split", "development")
    if split not in {"development", "eval"}:
        raise ValueError("Unknown packet split")
    if split == "eval":
        if packet.get("label_status") != "ai_reviewed_frozen_pre_target_query":
            raise ValueError("Evaluation labels must be frozen before target-model query")
        manifest_path = ROOT / source["selection_manifest_path"]
        if _text_sha256(manifest_path) != source["selection_manifest_sha256"]:
            raise ValueError("Selection manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["pdf_sha256"] != source["pdf_sha256"] or manifest["selected_count"] != len(candidates):
            raise ValueError("Selection manifest does not match source packet")
        if candidates_path.resolve() != (manifest_path.parent / "selected_passages.jsonl").resolve() or pages_path.resolve() != (manifest_path.parent / "source_pages.jsonl").resolve():
            raise ValueError("Evaluation source paths differ from frozen selection")
        frozen = manifest["frozen_artifact_sha256"]
        frozen_paths = {
            "selected_passages_sha256": candidates_path,
            "source_pages_sha256": pages_path,
            "doctrine_domain_sha256": ROOT / "data/ontology/doctrine_domain.kif",
            "sumo_classes_sha256": ROOT / "data/training_pairs/sumo_classes.jsonl",
            "sumo_relations_sha256": ROOT / "data/training_pairs/sumo_relations.jsonl",
            "relation_kinds_sha256": ROOT / "src/ontology/sumo_relation_kinds.json",
            "claim_contract_sha256": ROOT / "docs/CLAIM_CONTRACT.md",
        }
        for name, path in frozen_paths.items():
            if _text_sha256(path) != frozen[name]:
                raise ValueError(f"Frozen evaluation input changed: {name}")
    doctrine_axioms = set((ROOT / "data/ontology/doctrine_domain.kif").read_text(encoding="utf-8").splitlines())
    base_classes = load_closed_class_terms()
    seen = set()
    for row in packet["claims"]:
        rid = row["record_id"]
        if rid in seen or rid not in candidates:
            raise ValueError(f"duplicate or unknown record ID: {rid}")
        seen.add(rid)
        candidate = candidates[rid]
        if row["doc_id"] != candidate["doc_id"] or row["pdf_page"] != candidate["pdf_page"] or row["paragraph"] != candidate["paragraph"] or row["page"] != candidate["page"] or (candidate.get("edition") and row["edition"] != candidate["edition"]):
            raise ValueError(f"{rid}: source reference mismatch")
        span = row["source_span"]
        if not span or span not in candidate["source_excerpt"]:
            raise ValueError(f"{rid}: source span absent from paragraph excerpt")
        if _squash(span) not in _squash(pages[row["pdf_page"]]):
            raise ValueError(f"{rid}: source span absent from PDF page extraction")
        if row["status"] != "reviewed" or row["reviewer_type"] not in {"ai", "human"} or not row["reviewed_by"]:
            raise ValueError(f"{rid}: reviewer attribution missing")
        if row["split"] != split:
            raise ValueError(f"{rid}: row split differs from packet split")
        if split == "eval" and row["reviewer_type"] != "ai":
            raise ValueError(f"{rid}: this frozen evaluation packet is AI-reviewed only")
        decision = row["decision"]
        if decision == "abstain":
            if row.get("abstention_reason") not in _ABSTENTION_REASONS or not row.get("abstention_rationale"):
                raise ValueError(f"{rid}: abstention reason missing")
            if any(row.get(key) for key in ("claim_text", "cnl", "kif", "terms", "declarations", "coverage")):
                raise ValueError(f"{rid}: abstention cannot contain a proposed assertion")
            continue
        if decision != "positive":
            raise ValueError(f"{rid}: unknown decision")
        if row["coverage"] not in {"full", "partial"}:
            raise ValueError(f"{rid}: invalid coverage")
        if not row["claim_text"] or not row["entailment_rationale"] or not isinstance(row["omitted_qualifiers"], list):
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
    if split == "eval" and seen != set(candidates):
        raise ValueError("Evaluation labels must cover every selected source passage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        validate(packet)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"Claim packet invalid: {exc}\n")
    positives = sum(row["decision"] == "positive" for row in packet["claims"])
    print(f"Validated {positives} positive and {len(packet['claims']) - positives} abstention {packet.get('split', 'development')} rows; entailment is AI-reviewed, not mechanically proved")


if __name__ == "__main__":
    main()
