from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from scripts.prepare_training_data import is_military_focus_pair, prepare_training_data
from src.training.train import PATTERN_COVERAGE_SENTENCES


def _pair(pattern: str, index: int) -> dict[str, str]:
    return {
        "nl": f"{pattern} nl {index}",
        "cnl": f"{pattern} cnl {index}",
        "kif": f"({pattern} {index})",
        "pattern": pattern,
    }


def _make_scratch_dir() -> Path:
    path = Path(".pytest_tmp_scripts") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class TestPrepareTrainingData:
    def test_output_file_is_written(self):
        generator_batches = [
            ("instance", [_pair("instance", index) for index in range(6)]),
            ("conditional", [_pair("conditional", index) for index in range(6)]),
            ("negation", [_pair("negation", index) for index in range(6)]),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            output_path = scratch_dir / "train_balanced.jsonl"

            result = prepare_training_data(
                limit=12,
                output_path=output_path,
                generator_batches=generator_batches,
            )

            assert output_path.exists()
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert rows == result.pairs
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_per_pattern_counts_are_balanced(self):
        generator_batches = [
            ("instance", [_pair("instance", index) for index in range(9)]),
            ("conditional", [_pair("conditional", index) for index in range(5)]),
            ("conditional", [_pair("conditional", index + 10) for index in range(5)]),
            ("negation", [_pair("negation", index) for index in range(8)]),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=12,
                output_path=scratch_dir / "train_balanced.jsonl",
                generator_batches=generator_batches,
            )

            assert result.written_counts == {
                "instance": 4,
                "conditional": 4,
                "negation": 4,
            }
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_each_pattern_has_at_most_limit_div_n_patterns_pairs(self):
        generator_batches = [
            ("instance", [_pair("instance", index) for index in range(10)]),
            ("subclass", [_pair("subclass", index) for index in range(10)]),
            ("binary", [_pair("binary", index) for index in range(10)]),
            ("nary", [_pair("nary", index) for index in range(10)]),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=12,
                output_path=scratch_dir / "train_balanced.jsonl",
                generator_batches=generator_batches,
            )

            per_pattern_cap = 12 // len(result.generated_counts)

            for count in result.written_counts.values():
                assert count <= per_pattern_cap
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_excludes_exact_gold_and_probe_overlaps(self):
        probe_nl = PATTERN_COVERAGE_SENTENCES[0][1]
        generator_batches = [
            (
                "instance",
                [
                    {
                        "nl": "gold nl",
                        "cnl": "instance cnl keep",
                        "kif": "(instance keep)",
                        "pattern": "instance",
                    },
                    {
                        "nl": "safe nl",
                        "cnl": "gold cnl",
                        "kif": "(instance gold-cnl)",
                        "pattern": "instance",
                    },
                    {
                        "nl": probe_nl,
                        "cnl": "instance cnl probe",
                        "kif": "(instance probe)",
                        "pattern": "instance",
                    },
                    {
                        "nl": "clean instance",
                        "cnl": "instance cnl clean",
                        "kif": "(instance clean)",
                        "pattern": "instance",
                    },
                ],
            ),
            (
                "negation",
                [
                    {
                        "nl": "clean negation",
                        "cnl": "negation cnl clean",
                        "kif": "(not clean)",
                        "pattern": "negation",
                    }
                ],
            ),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=10,
                output_path=scratch_dir / "train_balanced.jsonl",
                generator_batches=generator_batches,
                excluded_nls={"gold nl"},
                excluded_cnls={"gold cnl"},
            )

            assert [pair["nl"] for pair in result.pairs] == ["clean instance", "clean negation"]
            assert result.excluded_counts == {
                "gold_nl": 1,
                "gold_cnl": 1,
                "probe_nl": 1,
            }
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_prioritizes_military_focus_pairs_within_pattern_cap(self):
        generator_batches = [
            (
                "binary",
                [
                    {
                        "nl": "background relation",
                        "cnl": "married Person Person",
                        "kif": "(married Person Person)",
                        "pattern": "binary",
                    },
                    {
                        "nl": "military relation",
                        "cnl": "agent Battle AutonomousAgent",
                        "kif": "(agent Battle AutonomousAgent)",
                        "pattern": "binary",
                    },
                ],
            ),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=1,
                output_path=scratch_dir / "train_balanced.jsonl",
                generator_batches=generator_batches,
            )

            assert len(result.pairs) == 1
            assert result.pairs[0]["cnl"] == "agent Battle AutonomousAgent"
            assert is_military_focus_pair(result.pairs[0]) is True
            assert result.focused_counts == {"binary": 1}
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
