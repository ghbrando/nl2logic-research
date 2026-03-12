"""
extract_sumo.py — Unified SUMO term extractor for NL2Logic pipeline.

Single-pass extraction over all .kif files, producing:
  - sumo_classes.jsonl    : {"term": "Process", "doc": "..."}
  - sumo_relations.jsonl  : {"term": "agent", "arity": 2, "signature": {"1": "Process", "2": "Agent"}, "doc": "..."}

Usage:
    python extract_sumo.py --kif-dir data/sumo --output-dir data/sumo

Patterns matched:
    Classes:    (subclass Foo Bar)           -> Foo AND Bar are classes
                (instance Foo Bar)           -> Foo is a class (capitalized first arg)
    Relations:  (domain rel N Type)          -> rel is a relation, arg N has type Type
                (range rel Type)             -> rel is a relation, output type is Type
                (instance rel *Predicate)    -> rel is a relation (meta-type declaration)
                (domainSubclass rel N Type)  -> rel is a relation, arg N constrained to subclass of Type
                (rangeSubclass rel Type)     -> rel is a relation, output constrained to subclass of Type
                (subrelation child parent)   -> child is a relation; arity inherited from parent in post-pass
    Docs:       (documentation Term EnglishLanguage "...")  -> doc string for Term
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns for S-expression statements
# ---------------------------------------------------------------------------

# Classes via subclass: both child AND parent are classes
RE_SUBCLASS = re.compile(
    r'^\s*\(\s*subclass\s+([A-Z][A-Za-z0-9_-]*)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Classes via instance: capitalized first arg
RE_INSTANCE = re.compile(
    r'^\s*\(\s*instance\s+([A-Z][A-Za-z0-9_-]*)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Relations via domain: (domain relName argNum ClassName)
RE_DOMAIN = re.compile(
    r'^\s*\(\s*domain\s+([a-z][A-Za-z0-9_-]*)\s+(\d+)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Relations via range: (range relName ClassName)
RE_RANGE = re.compile(
    r'^\s*\(\s*range\s+([a-z][A-Za-z0-9_-]*)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Also catch relations declared via instance: (instance relName BinaryPredicate) etc.
# These are lowercase first arg with a capitalized meta-class second arg
RE_INSTANCE_REL = re.compile(
    r'^\s*\(\s*instance\s+([a-z][A-Za-z0-9_-]*)\s+([A-Z][A-Za-z0-9_-]*(?:Predicate|Relation|Function)[A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Relations via domainSubclass: (domainSubclass relName argNum ClassName)
# Same arity/signature logic as domain; the subclass constraint is recorded in the signature.
RE_DOMAIN_SUBCLASS = re.compile(
    r'^\s*\(\s*domainSubclass\s+([a-z][A-Za-z0-9_-]*)\s+(\d+)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Relations via rangeSubclass: (rangeSubclass relName ClassName)
# Same logic as range; the subclass constraint is recorded in the signature.
RE_RANGE_SUBCLASS = re.compile(
    r'^\s*\(\s*rangeSubclass\s+([a-z][A-Za-z0-9_-]*)\s+([A-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Relations via subrelation: (subrelation child parent)
# The child is confirmed to be a relation; arity is inherited from parent in a post-pass.
RE_SUBRELATION = re.compile(
    r'^\s*\(\s*subrelation\s+([a-z][A-Za-z0-9_-]*)\s+([a-zA-Z][A-Za-z0-9_-]*)\s*\)',
    re.MULTILINE,
)

# Documentation: (documentation TermName EnglishLanguage "the doc string")
# Doc string may span multiple lines, so we use DOTALL.
# First pattern: strict (properly escaped quotes)
RE_DOC_STRICT = re.compile(
    r'\(\s*documentation\s+(\S+)\s+EnglishLanguage\s+"((?:[^"\\]|\\.)*)"\s*\)',
    re.DOTALL,
)
# Fallback pattern: greedy match to closing ")  — handles unescaped internal quotes
RE_DOC_GREEDY = re.compile(
    r'\(\s*documentation\s+(\S+)\s+EnglishLanguage\s+"(.+?)"\s*\)',
    re.DOTALL,
)


def _ensure_relation(relations: dict, rel_name: str) -> dict:
    """Return the relation record, creating it if absent."""
    if rel_name not in relations:
        relations[rel_name] = {"term": rel_name, "signature": {}}
    return relations[rel_name]


def _update_arity(record: dict, arg_num: int) -> None:
    """Set arity to arg_num if it is larger than the current value."""
    if arg_num > record.get("arity", 0):
        record["arity"] = arg_num


def extract_from_file(
    filepath: str,
    classes: dict,
    relations: dict,
    docs: dict,
    subrel_parents: dict,
) -> None:
    """Parse a single .kif file and accumulate into shared dicts.

    subrel_parents maps child relation name -> parent relation name for
    all (subrelation child parent) statements encountered. Arity inheritance
    is resolved in a separate post-pass after all files are processed.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # --- Classes via subclass (both child and parent) ---
    for m in RE_SUBCLASS.finditer(text):
        classes[m.group(1)] = True
        classes[m.group(2)] = True

    # --- Classes via instance (capitalized terms) ---
    for m in RE_INSTANCE.finditer(text):
        classes[m.group(1)] = True

    # --- Relations explicitly declared as Predicate/Relation/Function ---
    for m in RE_INSTANCE_REL.finditer(text):
        rel_name = m.group(1)
        rel_type = m.group(2)
        rec = _ensure_relation(relations, rel_name)
        if "type" not in rec:
            rec["type"] = rel_type

    # --- Relations from domain statements ---
    for m in RE_DOMAIN.finditer(text):
        rel_name = m.group(1)
        arg_num = int(m.group(2))
        arg_type = m.group(3)
        rec = _ensure_relation(relations, rel_name)
        rec["signature"][arg_num] = arg_type
        _update_arity(rec, arg_num)

    # --- Relations from domainSubclass statements (same arity logic as domain) ---
    for m in RE_DOMAIN_SUBCLASS.finditer(text):
        rel_name = m.group(1)
        arg_num = int(m.group(2))
        arg_type = m.group(3)
        rec = _ensure_relation(relations, rel_name)
        # Only fill signature slot if not already set by a domain statement
        if arg_num not in rec["signature"]:
            rec["signature"][arg_num] = arg_type
        _update_arity(rec, arg_num)

    # --- Relations from range statements ---
    for m in RE_RANGE.finditer(text):
        rel_name = m.group(1)
        arg_type = m.group(2)
        rec = _ensure_relation(relations, rel_name)
        rec["signature"]["range"] = arg_type

    # --- Relations from rangeSubclass statements (same logic as range) ---
    for m in RE_RANGE_SUBCLASS.finditer(text):
        rel_name = m.group(1)
        arg_type = m.group(2)
        rec = _ensure_relation(relations, rel_name)
        if "range" not in rec["signature"]:
            rec["signature"]["range"] = arg_type

    # --- Relations from subrelation statements ---
    # Register child as a relation now; arity is resolved in the post-pass.
    for m in RE_SUBRELATION.finditer(text):
        child = m.group(1)
        parent = m.group(2)
        _ensure_relation(relations, child)
        # Record the most recently seen parent (stable across files for SUMO)
        subrel_parents[child] = parent

    # --- Documentation strings (strict pass first, greedy fallback) ---
    for m in RE_DOC_STRICT.finditer(text):
        term = m.group(1)
        doc_text = m.group(2).strip()
        doc_text = re.sub(r'\s+', ' ', doc_text)
        if term not in docs:
            docs[term] = doc_text

    for m in RE_DOC_GREEDY.finditer(text):
        term = m.group(1)
        if term not in docs:
            doc_text = m.group(2).strip()
            doc_text = re.sub(r'\s+', ' ', doc_text)
            docs[term] = doc_text


def resolve_subrelation_arities(relations: dict, subrel_parents: dict) -> list[str]:
    """Propagate arity from parent to child via the subrelation hierarchy.

    Iterates until no further assignments can be made (handles multi-level chains).
    Returns a list of child terms whose arity could not be resolved.
    """
    changed = True
    while changed:
        changed = False
        for child, parent in subrel_parents.items():
            rec = relations.get(child)
            if rec is None or rec.get("arity") is not None:
                continue  # already resolved or child unknown
            parent_rec = relations.get(parent)
            if parent_rec and parent_rec.get("arity") is not None:
                rec["arity"] = parent_rec["arity"]
                changed = True

    return [
        child for child, parent in subrel_parents.items()
        if relations.get(child) and relations[child].get("arity") is None
    ]


def write_classes(classes: dict, docs: dict, output_path: str) -> int:
    """Write sumo_classes.jsonl — one JSON object per line."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(classes.keys()):
            record = {"term": term}
            if term in docs:
                record["doc"] = docs[term]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_relations(relations: dict, docs: dict, output_path: str) -> int:
    """Write sumo_relations.jsonl — one JSON object per line."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(relations.keys()):
            record = relations[term].copy()
            # Normalize signature keys to strings for JSON
            record["signature"] = {
                str(k): v for k, v in record["signature"].items()
            }
            if term in docs:
                record["doc"] = docs[term]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract SUMO classes and relations from .kif files."
    )
    parser.add_argument(
        "--kif-dir",
        type=str,
        default="data/sumo",
        help="Directory containing .kif files (default: data/sumo)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sumo",
        help="Output directory for .jsonl files (default: data/sumo)",
    )
    args = parser.parse_args()

    kif_dir = Path(args.kif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kif_files = sorted(kif_dir.glob("*.kif"))
    if not kif_files:
        print(f"ERROR: No .kif files found in {kif_dir}")
        return

    print(f"Found {len(kif_files)} .kif files in {kif_dir}")

    classes: dict[str, bool] = {}      # term -> True (ordered set)
    relations: dict[str, dict] = {}    # term -> {term, arity, signature, type?}
    docs: dict[str, str] = {}          # term -> doc string
    subrel_parents: dict[str, str] = {}  # child -> parent for subrelation statements

    for kif_file in kif_files:
        extract_from_file(str(kif_file), classes, relations, docs, subrel_parents)
        print(f"  Processed: {kif_file.name}")

    # --- Post-pass: resolve arity for subrelation-only terms ---
    unresolved = resolve_subrelation_arities(relations, subrel_parents)
    resolved = len(subrel_parents) - len(unresolved)
    print(f"\n  Subrelation post-pass: {len(subrel_parents)} edges, "
          f"{resolved} resolved, {len(unresolved)} unresolved")
    if unresolved:
        print(f"    Unresolved (no arity): {sorted(unresolved)[:20]}"
              f"{'...' if len(unresolved) > 20 else ''}")

    # --- Fallback: any documented capitalized term not already extracted ---
    # These are real SUMO terms (e.g. NAICS codes, Functions, Attributes)
    # declared via instance patterns we didn't catch explicitly.
    fallback_count = 0
    for term in docs:
        if term[0].isupper() and term not in classes and term not in relations:
            classes[term] = True
            fallback_count += 1
    if fallback_count:
        print(f"\n  Fallback: added {fallback_count} documented terms to classes")

    # --- Write outputs ---
    classes_path = output_dir / "sumo_classes.jsonl"
    relations_path = output_dir / "sumo_relations.jsonl"

    n_classes = write_classes(classes, docs, str(classes_path))
    n_relations = write_relations(relations, docs, str(relations_path))

    # --- Summary ---
    docs_matched_classes = sum(1 for t in classes if t in docs)
    docs_matched_relations = sum(1 for t in relations if t in docs)

    print(f"\n{'='*40}")
    print(f"  Extraction Summary")
    print(f"{'='*40}")
    print(f"  Classes:    {n_classes:>6}  (with docs: {docs_matched_classes})")
    print(f"  Relations:  {n_relations:>6}  (with docs: {docs_matched_relations})")
    print(f"  Total docs: {len(docs):>6}")
    print(f"{'='*40}")
    print(f"\n  Outputs:")
    print(f"    {classes_path}")
    print(f"    {relations_path}")

    # --- Sanity check: spot-check key military doctrine relations ---
    spot_check = ["agent", "patient", "instrument", "destination", "origin", "member"]
    found = [r for r in spot_check if r in relations]
    missing = [r for r in spot_check if r not in relations]
    print(f"\n  Spot-check (military-relevant relations):")
    print(f"    Found:   {found}")
    if missing:
        print(f"    Missing: {missing}")
    else:
        print(f"    All present [OK]")


if __name__ == "__main__":
    main()