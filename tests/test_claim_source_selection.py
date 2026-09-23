"""Guard the unreviewed chapter 2 source holdout against accidental changes."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter2-2023-claim-eval-source"


def test_source_selection_and_ontology_are_fingerprinted():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    files = {
        "selected_passages_sha256": SOURCE / "selected_passages.jsonl",
        "source_pages_sha256": SOURCE / "source_pages.jsonl",
        "doctrine_domain_sha256": ROOT / "data/ontology/doctrine_domain.kif",
        "sumo_classes_sha256": ROOT / "data/training_pairs/sumo_classes.jsonl",
        "sumo_relations_sha256": ROOT / "data/training_pairs/sumo_relations.jsonl",
        "relation_kinds_sha256": ROOT / "src/ontology/sumo_relation_kinds.json",
        "claim_contract_sha256": ROOT / "docs/CLAIM_CONTRACT.md",
    }
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["frozen_artifact_sha256"][name]


def test_selected_passages_have_provenance_but_no_labels():
    rows = [json.loads(line) for line in (SOURCE / "selected_passages.jsonl").read_text(encoding="utf-8").splitlines()]
    pages = {r["pdf_page"]: " ".join(r["text"].split()) for r in map(json.loads, (SOURCE / "source_pages.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(rows) == 29
    assert len({row["record_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["page"] == f"2-{row['pdf_page'] - 44}"
        assert row["status"] == "source_selected_unreviewed"
        assert row["decision"] is None
        assert " ".join(row["source_excerpt"].split()) in pages[row["pdf_page"]]
