"""Prepare unlabeled, source-traceable FM-style passages for human review."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.comparison import fingerprint, normalized_nl


def phenomena(text: str) -> list[str]:
    patterns = {
        "definition": r"\b(?:is|are)\s+(?:a|an|the)\b",
        "negation": r"\b(?:not|no|never|neither)\b|n['’]t\b",
        "condition": r"\b(?:if|unless|when|only|except)\b",
        "modality": r"\b(?:must|should|may|can|shall|could|might)\b",
        "relation": r"\b(?:supports?|provides?|includes?|requires?|consists?|enables?|integrates?)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)] or ["other"]


def first_sentence(text: str) -> str | None:
    # Protect common abbreviations; this remains a draft extraction heuristic.
    protected = text
    for abbreviation in ("U.S.", "e.g.", "i.e."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "\u2024"))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", protected, maxsplit=1)
    first = parts[0].replace("\u2024", ".").strip()
    return first if len(first.split()) >= 6 and first.endswith((".", "!", "?")) else None


def page_candidates(text: str, *, pdf_page: int, chapter: int, doc_id: str) -> list[dict]:
    lines = text.splitlines()
    footer = next((line for line in reversed(lines) if "FM 2-0" in line), "")
    printed = re.search(rf"\b{chapter}-\d+\b", footer)
    records, paragraph, buffer = [], None, []

    def flush():
        if not paragraph or not buffer:
            return
        excerpt = " ".join(buffer)
        sentence = first_sentence(excerpt)
        if sentence:
            records.append({
                "record_id": f"{doc_id}-p{pdf_page}-{paragraph}", "doc_id": doc_id,
                "page": printed.group() if printed else None, "pdf_page": pdf_page,
                "section": f"Chapter {chapter}", "paragraph": paragraph,
                "nl": sentence, "source_excerpt": excerpt, "phenomena": phenomena(sentence),
                "formalizable": None, "pattern": None, "cnl": None, "kif": None,
                "terms": [], "sumo_mapping_confidence": None,
                "status": "draft", "split": "eval", "reviewed_by": None,
                "review_seconds": None,
                "notes": "Automatically extracted first sentence of a numbered paragraph. Verify text/page and annotate independently; surface tags are not gold labels.",
            })

    for line in lines:
        line = line.strip()
        match = re.match(rf"^({chapter}-\d+)\.\s+(.*)", line)
        if match:
            flush()
            paragraph, buffer = match.group(1), [match.group(2)]
        elif ("FM 2-0" in line and re.search(r"\b20\d\d\b", line)) or re.match(r"^(?:Chapter |SECTION |Figure \d|Table \d|[\u26ab\u2022\u25aa\u25cf])", line) or (line.isupper() and len(line) < 100):
            flush()
            paragraph, buffer = None, []
        elif paragraph and line:
            buffer.append(line)
    flush()
    return records


def select_candidates(records: list[dict], count: int, seed: int, excluded: set[str]) -> list[dict]:
    unique = {}
    for row in records:
        key = normalized_nl(row["nl"])
        if key not in excluded:
            unique.setdefault(key, row)
    pool = list(unique.values())
    random.Random(seed).shuffle(pool)
    selected, used = [], set()
    # Round-robin tags enrich rare constructions without assigning labels.
    while len(selected) < min(count, len(pool)):
        for tag in ("negation", "condition", "modality", "definition", "relation", "other"):
            row = next((r for r in pool if r["record_id"] not in used and tag in r["phenomena"]), None)
            if row:
                selected.append(row)
                used.add(row["record_id"])
            if len(selected) == count:
                break
    return sorted(selected, key=lambda r: (r["pdf_page"], int(r["paragraph"].split("-")[1])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--doc-id", required=True, help="Include document edition/year")
    parser.add_argument("--chapter", required=True, type=int)
    parser.add_argument("--first-page", required=True, type=int, help="One-based PDF page")
    parser.add_argument("--last-page", required=True, type=int, help="Inclusive PDF page")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", action="append", type=Path, default=[], help="Existing benchmark/training JSONL texts")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing review work is never overwritten")
    args = parser.parse_args(argv)
    if args.count <= 0 or args.first_page < 1 or args.last_page < args.first_page:
        parser.error("Invalid count or page range")
    import pdfplumber
    excluded = set()
    for path in args.exclude:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("nl"):
                    excluded.add(normalized_nl(row["nl"]))
    pool, pages = [], []
    with pdfplumber.open(args.pdf) as pdf:
        if args.last_page > len(pdf.pages):
            parser.error("Page range exceeds PDF length")
        for number in range(args.first_page, args.last_page + 1):
            text = pdf.pages[number - 1].extract_text() or ""
            pages.append({"pdf_page": number, "text": text})
            pool.extend(page_candidates(text, pdf_page=number, chapter=args.chapter, doc_id=args.doc_id))
    selected = select_candidates(pool, args.count, args.seed, excluded)
    if len(selected) < args.count:
        parser.exit(2, f"Only {len(selected)} eligible passages available; requested {args.count}. No output written.\n")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("candidates.jsonl", selected), ("source_pages.jsonl", pages)):
        (args.output_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (args.output_dir / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["record_id", "pdf_page", "paragraph", "nl", "formalizable", "pattern", "cnl", "notes", "status", "reviewed_by", "review_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "source": fingerprint(args.pdf), "doc_id": args.doc_id, "chapter": args.chapter,
        "pdf_pages": [args.first_page, args.last_page], "seed": args.seed,
        "exclusions": [fingerprint(p) for p in args.exclude], "pool_count": len(pool),
        "selected_count": len(selected), "status": "draft_unscored",
        "pdfplumber_version": pdfplumber.__version__,
        "phenomena_counts": dict(Counter(tag for row in selected for tag in row["phenomena"])),
        "selection": "First complete sentence (at least six words) of numbered paragraphs; surface-tag round robin, seeded shuffle. Not a representative random sample.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
