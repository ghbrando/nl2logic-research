from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from scripts.check_leakage import check_leakage, main
from src.training.train import PATTERN_COVERAGE_SENTENCES


def _make_scratch_dir() -> Path:
    path = Path(".pytest_tmp_scripts") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class TestCheckLeakage:
    def test_clean_datasets_report_no_leakage_and_exit_zero(self, capsys):
        scratch_dir = _make_scratch_dir()
        try:
            train_path = scratch_dir / "train.jsonl"
            gold_path = scratch_dir / "gold.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"nl": "Train NL one.", "cnl": "?x is-a Process"},
                    {"nl": "Train NL two.", "cnl": "agent ?x ?y"},
                ],
            )
            _write_jsonl(
                gold_path,
                [
                    {"nl": "Gold NL one.", "cnl": "some ?x is-a Process", "kif": "...", "pattern": "existential"},
                ],
            )

            report = check_leakage(training_path=train_path, gold_path=gold_path)

            assert report.has_leakage is False
            assert main(["--train-path", str(train_path), "--gold-path", str(gold_path)]) == 0
            captured = capsys.readouterr()
            assert "Status: clean" in captured.out
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_exact_nl_match_is_detected(self):
        scratch_dir = _make_scratch_dir()
        try:
            train_path = scratch_dir / "train.jsonl"
            gold_path = scratch_dir / "gold.jsonl"
            shared_nl = "Shared doctrine sentence."
            _write_jsonl(train_path, [{"nl": shared_nl, "cnl": "?x is-a Process"}])
            _write_jsonl(
                gold_path,
                [{"nl": shared_nl, "cnl": "some ?x is-a Process", "kif": "...", "pattern": "existential"}],
            )

            report = check_leakage(training_path=train_path, gold_path=gold_path)

            assert report.nl_matches == [("nl", shared_nl)]
            assert report.has_leakage is True
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_exact_cnl_match_is_detected(self):
        scratch_dir = _make_scratch_dir()
        try:
            train_path = scratch_dir / "train.jsonl"
            gold_path = scratch_dir / "gold.jsonl"
            shared_cnl = "?x is-a Process"
            _write_jsonl(train_path, [{"nl": "Train sentence.", "cnl": shared_cnl}])
            _write_jsonl(
                gold_path,
                [{"nl": "Gold sentence.", "cnl": shared_cnl, "kif": "...", "pattern": "instance"}],
            )

            report = check_leakage(training_path=train_path, gold_path=gold_path)

            assert report.cnl_matches == [("cnl", shared_cnl)]
            assert report.has_leakage is True
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_ignore_cnl_overlap_allows_semantic_paraphrase_training(self):
        scratch_dir = _make_scratch_dir()
        try:
            train_path = scratch_dir / "train.jsonl"
            gold_path = scratch_dir / "gold.jsonl"
            shared_cnl = "?x is-a Process"
            _write_jsonl(train_path, [{"nl": "Train sentence.", "cnl": shared_cnl}])
            _write_jsonl(
                gold_path,
                [{"nl": "Gold sentence.", "cnl": shared_cnl, "kif": "...", "pattern": "instance"}],
            )

            report = check_leakage(
                training_path=train_path,
                gold_path=gold_path,
                check_gold_cnl_overlap=False,
            )

            assert report.cnl_matches == []
            assert report.has_leakage is False
            assert main([
                "--train-path", str(train_path),
                "--gold-path", str(gold_path),
                "--ignore-cnl-overlap",
            ]) == 0
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_probe_sentence_match_is_detected(self):
        scratch_dir = _make_scratch_dir()
        try:
            train_path = scratch_dir / "train.jsonl"
            gold_path = scratch_dir / "gold.jsonl"
            probe_pattern, probe_nl = PATTERN_COVERAGE_SENTENCES[0]
            _write_jsonl(train_path, [{"nl": probe_nl, "cnl": "?x is-a Process"}])
            _write_jsonl(
                gold_path,
                [{"nl": "Gold sentence.", "cnl": "some ?x is-a Process", "kif": "...", "pattern": "existential"}],
            )

            report = check_leakage(training_path=train_path, gold_path=gold_path)

            assert report.probe_matches == [(probe_pattern, probe_nl)]
            assert main(["--train-path", str(train_path), "--gold-path", str(gold_path)]) == 1
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
