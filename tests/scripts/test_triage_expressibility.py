from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.triage_expressibility import BLOCKERS, TRIAGE, VERDICTS, build, main, summarize

PACKET = Path("data/benchmarks/fm2-0/chapter1-2023-review/candidates.jsonl")


def _candidate(record_id: str, nl: str = "A weapon is an artifact.", **overrides) -> dict:
    record = {
        "record_id": record_id,
        "pdf_page": 18,
        "paragraph": "1-3",
        "nl": nl,
        "phenomena": ["definition"],
    }
    record.update(overrides)
    return record


def test_every_verdict_is_well_formed() -> None:
    for record_id, (verdict, blockers, _pattern, note) in TRIAGE.items():
        assert verdict in VERDICTS, record_id
        assert note, f"{record_id} has no note"
        for code in blockers:
            assert code in BLOCKERS, f"{record_id} uses unknown blocker {code}"


def test_blocked_records_always_name_a_blocker() -> None:
    for record_id, (verdict, blockers, pattern, _note) in TRIAGE.items():
        if verdict == "blocked":
            assert blockers, f"{record_id} is blocked but names no blocker"
            assert not pattern, f"{record_id} is blocked but proposes pattern {pattern}"


def test_expressible_and_partial_records_propose_a_pattern() -> None:
    for record_id, (verdict, _blockers, pattern, _note) in TRIAGE.items():
        if verdict in {"expressible", "partial"}:
            assert pattern, f"{record_id} is {verdict} but proposes no pattern"


def test_partial_records_always_record_a_loss() -> None:
    """A partial verdict without a named loss is indistinguishable from expressible."""
    for record_id, (verdict, blockers, _pattern, note) in TRIAGE.items():
        if verdict == "partial":
            assert blockers or note, f"{record_id} is partial but records no loss"


def test_build_rejects_unknown_candidate() -> None:
    with pytest.raises(SystemExit, match="no triage verdict"):
        build([_candidate("not-a-real-record")])


def test_build_rejects_duplicate_record_ids() -> None:
    known = next(iter(TRIAGE))
    with pytest.raises(SystemExit, match="duplicate record_id"):
        build([_candidate(known), _candidate(known)])


def test_build_rejects_triage_without_matching_candidates() -> None:
    known = next(iter(TRIAGE))
    with pytest.raises(SystemExit, match="triage references"):
        build([_candidate(known)])


def test_summary_partitions_the_packet() -> None:
    rows = build([json.loads(line) for line in PACKET.read_text(encoding="utf-8").splitlines() if line.strip()])
    summary = summarize(rows)
    assert summary["total"] == 100
    assert sum(summary["verdicts"].values()) == 100
    assert summary["expressible_substantive"] + summary["expressible_bibliographic"] == summary["verdicts"]["expressible"]


def test_bibliographic_expressibles_are_separated_from_substantive() -> None:
    """Bibliographic rows compile but answer no useful question; they must not inflate the ceiling."""
    rows = build([json.loads(line) for line in PACKET.read_text(encoding="utf-8").splitlines() if line.strip()])
    summary = summarize(rows)
    assert summary["expressible_bibliographic"] > 0
    assert summary["expressible_substantive"] < summary["verdicts"]["expressible"]


def test_main_writes_reviewable_csv_and_summary(tmp_path: Path) -> None:
    out_csv = tmp_path / "triage.csv"
    out_json = tmp_path / "summary.json"
    assert main(["--candidates", str(PACKET), "--out-csv", str(out_csv), "--out-json", str(out_json)]) == 0

    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 101
    assert "ceiling_verdict" in lines[0]
    assert "claude-draft-triage" in lines[1]

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 100
    assert set(payload["blocker_glossary"]) == set(BLOCKERS)


def test_triage_output_is_not_a_gold_annotation_file(tmp_path: Path) -> None:
    """finalize_benchmark_review.py must never be able to promote these drafts."""
    out_csv = tmp_path / "triage.csv"
    out_json = tmp_path / "summary.json"
    main(["--candidates", str(PACKET), "--out-csv", str(out_csv), "--out-json", str(out_json)])

    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "formalizable" not in header
    assert "cnl" not in header.split(",")
    assert "reviewed_by" not in header
