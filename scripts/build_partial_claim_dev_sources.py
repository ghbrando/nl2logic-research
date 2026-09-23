"""Project reviewed development IDs onto their original, unlabeled source text."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import text_sha256
from scripts.validate_claim_packet import validate


def build(packet_path: Path, candidates_path: Path, output_dir: Path) -> None:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    validate(packet)
    if any(row["split"] != "development" for row in packet["claims"]):
        raise ValueError("This source projection is for development data only")
    by_id = {row["record_id"]: row for row in map(json.loads, candidates_path.read_text(encoding="utf-8").splitlines())}
    rows = []
    for claim in packet["claims"]:
        source = by_id[claim["record_id"]]
        sentence = source["nl"]
        excerpt = source["source_excerpt"]
        if not excerpt.startswith(sentence):
            raise ValueError(f"{claim['record_id']}: first sentence is not exact paragraph prefix")
        rows.append({key: source[key] for key in (
            "record_id", "doc_id", "page", "pdf_page", "paragraph", "source_excerpt"
        )} | {"source_sentence": sentence})
    output_dir.mkdir(parents=True, exist_ok=False)
    source_path = output_dir / "sources.jsonl"
    source_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "purpose": "Unlabeled chapter 1 development source projection; selection IDs came from an AI-reviewed development packet",
        "packet_sha256": text_sha256(packet_path),
        "candidates_sha256": text_sha256(candidates_path),
        "sources_sha256": text_sha256(source_path),
        "source_count": len(rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.packet, args.candidates, args.output_dir)


if __name__ == "__main__":
    main()
