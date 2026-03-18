from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.build_domain import (
    DomainBuildError,
    build_metadata,
    load_accepted_records,
    render_theory,
    run_vampire_validation,
    write_domain_bundle,
)


def _write_accepted(path: Path) -> None:
    rows = [
        {
            "record_id": "p1-pos0-n0",
            "kif": "(agent MilitaryProcess AutonomousAgent)",
            "cnl": "agent MilitaryProcess AutonomousAgent",
            "nl": "A military process has an autonomous agent.",
            "original": "A military process has an autonomous agent.",
            "page_number": 1,
            "position": 0,
            "section": "TASKS",
            "relation": "agent",
            "terms": ["agent", "MilitaryProcess", "AutonomousAgent"],
        },
        {
            "record_id": "p2-pos1-n0",
            "kif": "(destination Transportation Region)",
            "cnl": "destination Transportation Region",
            "nl": "A transportation process has a region as its destination.",
            "original": "A transportation process has a region as its destination.",
            "page_number": 2,
            "position": 1,
            "section": "MOVEMENT",
            "relation": "destination",
            "terms": ["destination", "Transportation", "Region"],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_accepted_records_reads_jsonl(tmp_path: Path):
    accepted_path = tmp_path / "accepted.jsonl"
    _write_accepted(accepted_path)

    records = load_accepted_records(accepted_path)

    assert [record["record_id"] for record in records] == ["p1-pos0-n0", "p2-pos1-n0"]


def test_load_accepted_records_rejects_duplicate_record_ids(tmp_path: Path):
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        '{"record_id":"dup","kif":"(foo)"}\n{"record_id":"dup","kif":"(bar)"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DomainBuildError, match="Duplicate record_id"):
        load_accepted_records(accepted_path)


def test_render_theory_and_metadata_include_provenance(tmp_path: Path):
    accepted_path = tmp_path / "accepted.jsonl"
    _write_accepted(accepted_path)
    records = load_accepted_records(accepted_path)

    theory_text = render_theory(records, ["sumo/Core.kif"])
    metadata = build_metadata(records, ["sumo/Core.kif"])

    assert ";;; Referenced SUMO sources:" in theory_text
    assert ";;; p1-pos0-n0 | page 1 | section TASKS" in theory_text
    assert "(destination Transportation Region)" in theory_text
    assert metadata["sumo_sources"] == ["sumo/Core.kif"]
    assert metadata["records"]["p2-pos1-n0"]["section"] == "MOVEMENT"


def test_write_domain_bundle_writes_theory_and_metadata_files(tmp_path: Path):
    accepted_path = tmp_path / "accepted.jsonl"
    theory_path = tmp_path / "domain.kif"
    metadata_path = tmp_path / "domain_metadata.json"
    _write_accepted(accepted_path)
    records = load_accepted_records(accepted_path)

    write_domain_bundle(
        records,
        theory_path=theory_path,
        metadata_path=metadata_path,
        sumo_paths=["sumo/Core.kif"],
    )

    assert theory_path.exists()
    assert metadata_path.exists()
    assert "(agent MilitaryProcess AutonomousAgent)" in theory_path.read_text(encoding="utf-8")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 2


def test_run_vampire_validation_invokes_subprocess(monkeypatch, tmp_path: Path):
    theory_path = tmp_path / "domain.kif"
    theory_path.write_text("(agent MilitaryProcess AutonomousAgent)\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, check, capture_output, text):
        captured["command"] = command
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        return subprocess.CompletedProcess(command, 0, stdout="success", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_vampire_validation("vampire", theory_path)

    assert captured["command"] == ["vampire", str(theory_path)]
    assert result.returncode == 0
    assert result.stdout == "success"
