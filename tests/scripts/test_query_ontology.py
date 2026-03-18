from __future__ import annotations

import json
from pathlib import Path

from scripts.ingest_pipeline import KifRecord, build_query_index
from scripts.query_ontology import format_record, load_index, query_records


def _write_index(path: Path) -> dict:
    records = [
        KifRecord(
            record_id="p1-pos0-n0",
            kif="(agent MilitaryProcess AutonomousAgent)",
            cnl="agent MilitaryProcess AutonomousAgent",
            nl="A military process has an autonomous agent.",
            original="A military process has an autonomous agent.",
            page_number=1,
            position=0,
            section="TASKS",
            pattern="binary",
            relation="agent",
            terms=["agent", "MilitaryProcess", "AutonomousAgent"],
        ),
        KifRecord(
            record_id="p2-pos1-n0",
            kif="(destination Transportation Region)",
            cnl="destination Transportation Region",
            nl="A transportation process has a region as its destination.",
            original="A transportation process has a region as its destination.",
            page_number=2,
            position=1,
            section="MOVEMENT",
            pattern="binary",
            relation="destination",
            terms=["destination", "Transportation", "Region"],
        ),
    ]
    index = build_query_index(records)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def test_load_index_reads_json_payload(tmp_path: Path):
    index_path = tmp_path / "index.json"
    expected = _write_index(index_path)

    loaded = load_index(index_path)

    assert loaded == expected


def test_query_records_applies_and_filters_using_index_maps(tmp_path: Path):
    index_path = tmp_path / "index.json"
    index = _write_index(index_path)

    matches = query_records(index, relation="destination", term="Region", section="MOVEMENT", page=2, limit=10)

    assert [record["record_id"] for record in matches] == ["p2-pos1-n0"]


def test_query_records_text_filter_matches_case_insensitively(tmp_path: Path):
    index_path = tmp_path / "index.json"
    index = _write_index(index_path)

    matches = query_records(index, text="autonomous AGENT", limit=10)

    assert [record["record_id"] for record in matches] == ["p1-pos0-n0"]


def test_format_record_includes_core_fields():
    record = {
        "record_id": "p1-pos0-n0",
        "page_number": 1,
        "section": "TASKS",
        "relation": "agent",
        "nl": "A military process has an autonomous agent.",
        "cnl": "agent MilitaryProcess AutonomousAgent",
        "kif": "(agent MilitaryProcess AutonomousAgent)",
    }

    formatted = format_record(record, 1)

    assert "[1] p1-pos0-n0 | page 1 | TASKS" in formatted
    assert "Relation: agent" in formatted
    assert "NL: A military process has an autonomous agent." in formatted
