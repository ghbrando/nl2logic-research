from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_ontology_leakage import audit, main, read_targets, render

ONTOLOGY = "\n".join([
    "(subclass Alpha Thing)",
    "(documentation Alpha EnglishLanguage \"ignored\")",
    "(subclass Beta Thing)",
    "(subclass Gamma Thing)",
    ";; (subclass Commented Thing)",
])


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def ontology(tmp_path: Path) -> Path:
    path = tmp_path / "doctrine.kif"
    path.write_text(ONTOLOGY, encoding="utf-8")
    return path


def test_read_targets_skips_records_without_a_target(tmp_path: Path) -> None:
    path = write(tmp_path / "b.jsonl", [
        {"kif": "(subclass Alpha Thing)"},
        {"kif": None},
        {"nl": "no kif field at all"},
    ])
    assert read_targets(path) == [("Alpha", "Thing")]


def test_read_targets_collects_every_claim_in_a_multi_claim_target(tmp_path: Path) -> None:
    path = write(tmp_path / "b.jsonl", [
        {"kif": "(subclass Alpha Thing)\n(subclass Beta Thing)"},
    ])
    assert read_targets(path) == [("Alpha", "Thing"), ("Beta", "Thing")]


def test_commented_assertions_are_not_counted(tmp_path: Path) -> None:
    report = audit(ontology(tmp_path), [], [write(tmp_path / "b.jsonl", [{"kif": "(subclass Commented Thing)"}])])
    assert report["ontology_subclass_assertions"] == 3
    assert report["benchmark_targets_outside_ontology"] == ["Commented Thing"]


def test_disjoint_split_from_one_pool_is_flagged(tmp_path: Path) -> None:
    """The exact-text check passes here; the audit is what catches it."""
    report = audit(
        ontology(tmp_path),
        [write(tmp_path / "t.jsonl", [{"kif": "(subclass Alpha Thing)"}])],
        [write(tmp_path / "b.jsonl", [{"kif": "(subclass Beta Thing)"}])],
    )
    assert report["ontology_used_by_both"] == 0
    assert report["split_partitions_one_ontology_pool"] is True
    assert "partition one ontology pool" in render(report)


def test_novel_benchmark_targets_clear_the_verdict(tmp_path: Path) -> None:
    report = audit(
        ontology(tmp_path),
        [write(tmp_path / "t.jsonl", [{"kif": "(subclass Alpha Thing)"}])],
        [write(tmp_path / "b.jsonl", [
            {"kif": "(subclass Delta Thing)"},
            {"kif": "(subclass Epsilon Thing)"},
        ])],
    )
    assert report["split_partitions_one_ontology_pool"] is False
    assert report["benchmark_files"][0]["novel"] == 2
    assert "2 benchmark target(s) are not asserted" in render(report)


def test_unused_assertions_are_reported(tmp_path: Path) -> None:
    report = audit(
        ontology(tmp_path),
        [write(tmp_path / "t.jsonl", [{"kif": "(subclass Alpha Thing)"}])],
        [write(tmp_path / "b.jsonl", [{"kif": "(subclass Beta Thing)"}])],
    )
    assert report["ontology_never_used"] == 1  # Gamma


def test_main_requires_at_least_one_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--ontology", str(ontology(tmp_path))])


def test_main_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--ontology", str(ontology(tmp_path)), "--benchmark", str(tmp_path / "nope.jsonl")])


def test_main_writes_json_report(tmp_path: Path) -> None:
    out = tmp_path / "report" / "audit.json"
    code = main([
        "--ontology", str(ontology(tmp_path)),
        "--training", str(write(tmp_path / "t.jsonl", [{"kif": "(subclass Alpha Thing)"}])),
        "--benchmark", str(write(tmp_path / "b.jsonl", [{"kif": "(subclass Beta Thing)"}])),
        "--json", str(out),
    ])
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ontology_subclass_assertions"] == 3


def test_real_doctrine_split_is_still_fully_restated() -> None:
    """Pins today's finding: the shipped doctrine split adds no novel targets.

    If this fails because novel targets appeared, that is progress - update the
    expectation. It must not fail silently.
    """
    report = audit(
        Path("data/ontology/doctrine_domain.kif"),
        [Path("data/training_pairs/doctrine_real_train.jsonl")],
        [Path("data/benchmarks/fm2-0/seed_positive.jsonl")],
    )
    assert report["training_files"][0]["novel"] == 0
    assert report["benchmark_files"][0]["novel"] == 0
    assert report["ontology_used_by_both"] == 0
