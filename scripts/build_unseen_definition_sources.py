"""Freeze unseen FM 2-0 definition passages from chapters not yet queried.

Selection is label-free and prediction-free: every numbered paragraph in
chapter 3 or later for which the frozen conservative extractor finds a copular
definition head without input-gate reasons. Paragraphs whose first sentence
overlaps any earlier development, evaluation, or seed source are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_benchmark_review import page_candidates
from scripts.validate_claim_packet import _text_sha256
from src.eval.comparison import normalized_nl
from src.preprocessing.partial_claims import extract_paragraph_candidate

FM = ROOT / "data/benchmarks/fm2-0"
EXCLUDE = [
    FM / "chapter1-2023-review/candidates.jsonl",
    FM / "chapter2-2023-claim-eval-source/selected_passages.jsonl",
    FM / "seed_positive.jsonl", FM / "seed_review.jsonl", FM / "seed_abstain.jsonl",
]
EDITION = "01 October 2023"
_PARAGRAPH = re.compile(r"^(\d+)-\d+\.\s", re.MULTILINE)


def build(pdf_path: Path, output_dir: Path, *, first_chapter: int = 3) -> dict:
    import pdfplumber

    excluded = set()
    for path in EXCLUDE:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            for key in ("nl", "source_first_sentence"):
                if row.get(key):
                    excluded.add(normalized_nl(row[key]))
    selected, pages, scanned, overlaps = [], {}, 0, 0
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            chapters = sorted({int(c) for c in _PARAGRAPH.findall(text) if int(c) >= first_chapter})
            for chapter in chapters:
                for row in page_candidates(text, pdf_page=number, chapter=chapter, doc_id="fm2-0-2023"):
                    scanned += 1
                    if normalized_nl(row["nl"]) in excluded:
                        overlaps += 1
                        continue
                    candidate, _ = extract_paragraph_candidate(row["source_excerpt"], row["nl"])
                    if candidate.status != "candidate" or candidate.gate_reasons:
                        continue
                    selected.append({key: row[key] for key in (
                        "record_id", "doc_id", "page", "pdf_page", "section", "paragraph", "nl", "source_excerpt"
                    )} | {"edition": EDITION})
                    pages[number] = text
    if len({row["record_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate record IDs in unseen selection")
    output_dir.mkdir(parents=True, exist_ok=False)

    def write(name: str, rows: list[dict]) -> Path:
        path = output_dir / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return path

    passages = write("selected_passages.jsonl", selected)
    page_rows = write("source_pages.jsonl", [{"pdf_page": n, "text": pages[n]} for n in sorted(pages)])
    sources = write("sources.jsonl", [
        {key: row[key] for key in ("record_id", "doc_id", "page", "pdf_page", "paragraph", "source_excerpt")}
        | {"source_sentence": row["nl"]} for row in selected
    ])
    manifest = {
        "source_pdf": pdf_path.relative_to(ROOT).as_posix(),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "doc_id": "fm2-0-2023",
        "edition": EDITION,
        "chapters": f">= {first_chapter} (numbered paragraphs only; appendices excluded)",
        "selected_count": len(selected),
        "source_count": len(selected),
        "scanned_paragraph_count": scanned,
        "excluded_first_sentence_overlap_count": overlaps,
        "selection_rule": "All numbered paragraphs in chapter 3 or later whose page_candidates first sentence has at least six words and for which extract_paragraph_candidate returns a candidate with no input-gate reasons; first-sentence overlaps with earlier sources excluded. No labels, predictions, or manual choice.",
        "extraction": "pdfplumber extract_text via scripts.prepare_benchmark_review.page_candidates",
        "status": "source_selected_unreviewed",
        "training_use": False,
        "selection_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources_sha256": _text_sha256(sources),
        "frozen_artifact_sha256": {
            "selected_passages_sha256": _text_sha256(passages),
            "source_pages_sha256": _text_sha256(page_rows),
            "doctrine_domain_sha256": _text_sha256(ROOT / "data/ontology/doctrine_domain.kif"),
            "sumo_classes_sha256": _text_sha256(ROOT / "data/training_pairs/sumo_classes.jsonl"),
            "sumo_relations_sha256": _text_sha256(ROOT / "data/training_pairs/sumo_relations.jsonl"),
            "relation_kinds_sha256": _text_sha256(ROOT / "src/ontology/sumo_relation_kinds.json"),
            "claim_contract_sha256": _text_sha256(ROOT / "docs/CLAIM_CONTRACT.md"),
        },
        "target_nl2logic_model_queries_after_selection": 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "src/data/ARMY FM_2-0.pdf")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.pdf.resolve(), args.output_dir)
    print(json.dumps({key: manifest[key] for key in (
        "selected_count", "scanned_paragraph_count", "excluded_first_sentence_overlap_count")}))


if __name__ == "__main__":
    main()
