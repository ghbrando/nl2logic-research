from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from scripts.prepare_training_data import (
    MILITARY_FOCUS_CLASSES,
    _REAL_DOCTRINE_TRAIN_PATH,
    is_military_focus_pair,
    load_benchmark_overlap_sets,
    load_real_doctrine_pairs,
    prepare_training_data,
)
from src.compiler.compiler import CNLCompiler
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

    def test_allow_gold_cnl_overlap_keeps_non_exact_nl_paraphrases(self):
        generator_batches = [
            (
                "existential",
                [
                    {
                        "nl": "There exists a plan for an operation.",
                        "cnl": "some ?x is-a Plan",
                        "kif": "(exists (?x) (instance ?x Plan))",
                        "pattern": "existential",
                    }
                ],
            ),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=4,
                output_path=scratch_dir / "train_balanced.jsonl",
                generator_batches=generator_batches,
                excluded_nls=set(),
                excluded_cnls=None,
                exclude_gold_cnl_overlap=False,
            )

            assert [pair["nl"] for pair in result.pairs] == ["There exists a plan for an operation."]
            assert result.excluded_counts["gold_cnl"] == 0
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)


class TestDoctrineTermPrioritization:
    """Doctrine domain terms must be recognized as military-focus and survive balancing."""

    _DOCTRINE_TERMS = [
        "CombatInformation",
        "DoctrinalTask",
        "GeospatialIntelligence",
        "HumanIntelligence",
        "InformationCollection",
        "IntelligenceDiscipline",
        "IntelligenceDissemination",
        "IntelligenceEnterprise",
        "IntelligenceProcess",
        "IntelligenceProduct",
        "IntelligenceProfessional",
        "IntelligenceWarfightingFunction",
        "IntelligenceWarfightingFunctionTask",
        "OperationalEnvironment",
        "SignalsIntelligence",
        "TacticalCommander",
        "ThreatCourseOfAction",
        "WarfightingFunction",
    ]

    def test_all_doctrine_terms_in_focus_set(self):
        for term in self._DOCTRINE_TERMS:
            assert term in MILITARY_FOCUS_CLASSES, (
                f"Doctrine term '{term}' missing from MILITARY_FOCUS_CLASSES"
            )

    def test_doctrine_pair_recognized_as_military_focus(self):
        pair = {
            "nl": "HumanIntelligence is a type of IntelligenceDiscipline.",
            "cnl": "HumanIntelligence subclass-of IntelligenceDiscipline",
            "kif": "(subclass HumanIntelligence IntelligenceDiscipline)",
            "pattern": "subclass",
        }
        assert is_military_focus_pair(pair) is True

    def test_doctrine_pair_prioritized_over_non_focus(self):
        """A doctrine pair wins the slot when competing with a non-focus pair at the cap."""
        non_focus = {
            "nl": "Vertebrate is a type of Animal.",
            "cnl": "Vertebrate subclass-of Animal",
            "kif": "(subclass Vertebrate Animal)",
            "pattern": "subclass",
        }
        # Use InformationCollection — a doctrine term NOT in the benchmark CNL set.
        doctrine_pair = {
            "nl": "InformationCollection is a type of MilitaryProcess.",
            "cnl": "InformationCollection subclass-of MilitaryProcess",
            "kif": "(subclass InformationCollection MilitaryProcess)",
            "pattern": "subclass",
        }
        generator_batches = [
            ("subclass", [non_focus, doctrine_pair]),
        ]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=1,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
            )
            assert len(result.pairs) == 1
            assert result.pairs[0]["cnl"] == "InformationCollection subclass-of MilitaryProcess"
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_doctrine_pairs_preserved_in_balanced_output(self):
        """All doctrine pairs fit within cap when there are no competing non-focus pairs."""
        # Use terms NOT in the benchmark CNL set so they survive exclusion filtering.
        doctrine_pairs = [
            {
                "nl": f"{child} is a type of {parent}.",
                "cnl": f"{child} subclass-of {parent}",
                "kif": f"(subclass {child} {parent})",
                "pattern": "subclass",
            }
            for child, parent in [
                ("WarfightingFunction", "MilitaryProcess"),
                ("InformationCollection", "MilitaryProcess"),
                ("IntelligenceDissemination", "Communication"),
            ]
        ]
        generator_batches = [("subclass", doctrine_pairs)]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=3,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
            )
            written_cnls = {p["cnl"] for p in result.pairs}
            assert "WarfightingFunction subclass-of MilitaryProcess" in written_cnls
            assert "InformationCollection subclass-of MilitaryProcess" in written_cnls
            assert "IntelligenceDissemination subclass-of Communication" in written_cnls
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)


class TestBenchmarkExclusionAndDoctrinePreservation:
    """FM 2-0 benchmark rows must be excluded; real-doctrine rows must be preserved."""

    # One NL from each benchmark split to verify exclusion coverage.
    _BENCHMARK_POSITIVE_NL = (
        "Human intelligence is the collection by a trained human intelligence collector "
        "of foreign information from people and multimedia to identify elements, intentions, "
        "composition, strength, dispositions, tactics, equipment, and capabilities (ADP 2-0)."
    )
    _BENCHMARK_REVIEW_NL = (
        "Combat information is a report that is gathered by or provided to the tactical commander."
    )
    _BENCHMARK_POSITIVE_CNL = "HumanIntelligence subclass-of IntelligenceDiscipline"
    _BENCHMARK_REVIEW_CNL = "IntelligenceProcess subclass-of MilitaryProcess"

    def test_load_benchmark_overlap_sets_contains_positive_nl(self):
        bm_nls, _ = load_benchmark_overlap_sets()
        assert self._BENCHMARK_POSITIVE_NL in bm_nls

    def test_load_benchmark_overlap_sets_contains_review_nl(self):
        bm_nls, _ = load_benchmark_overlap_sets()
        assert self._BENCHMARK_REVIEW_NL in bm_nls

    def test_load_benchmark_overlap_sets_contains_positive_cnl(self):
        _, bm_cnls = load_benchmark_overlap_sets()
        assert self._BENCHMARK_POSITIVE_CNL in bm_cnls

    def test_load_benchmark_overlap_sets_contains_review_cnl(self):
        _, bm_cnls = load_benchmark_overlap_sets()
        assert self._BENCHMARK_REVIEW_CNL in bm_cnls

    def test_benchmark_nl_excluded_from_prepared_output(self):
        """A pair whose NL is a benchmark sentence must not appear in the output."""
        benchmark_pair = {
            "nl": self._BENCHMARK_POSITIVE_NL,
            "cnl": "HumanIntelligence subclass-of IntelligenceDiscipline",
            "kif": "(subclass HumanIntelligence IntelligenceDiscipline)",
            "pattern": "subclass",
        }
        safe_pair = {
            "nl": "InformationCollection is a type of MilitaryProcess.",
            "cnl": "InformationCollection subclass-of MilitaryProcess",
            "kif": "(subclass InformationCollection MilitaryProcess)",
            "pattern": "subclass",
        }
        generator_batches = [("subclass", [benchmark_pair, safe_pair])]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=2,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
                exclude_benchmark_overlap=True,
            )
            output_nls = {p["nl"] for p in result.pairs}
            assert self._BENCHMARK_POSITIVE_NL not in output_nls
            assert "InformationCollection is a type of MilitaryProcess." in output_nls
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_benchmark_cnl_excluded_from_prepared_output(self):
        """A pair whose CNL is a benchmark CNL must not appear in the output."""
        benchmark_cnl_pair = {
            "nl": "An intelligence process is a kind of military process.",
            "cnl": self._BENCHMARK_REVIEW_CNL,
            "kif": "(subclass IntelligenceProcess MilitaryProcess)",
            "pattern": "subclass",
        }
        safe_pair = {
            "nl": "InformationCollection is a type of MilitaryProcess.",
            "cnl": "InformationCollection subclass-of MilitaryProcess",
            "kif": "(subclass InformationCollection MilitaryProcess)",
            "pattern": "subclass",
        }
        generator_batches = [("subclass", [benchmark_cnl_pair, safe_pair])]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=2,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
                exclude_benchmark_overlap=True,
            )
            output_cnls = {p["cnl"] for p in result.pairs}
            assert self._BENCHMARK_REVIEW_CNL not in output_cnls
            assert "InformationCollection subclass-of MilitaryProcess" in output_cnls
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_pinned_pairs_are_preserved_in_output(self):
        """Real-doctrine pinned pairs survive balancing and appear in the output."""
        pinned = [
            {
                "nl": "A warfighting function is a group of tasks and systems unified by a common purpose.",
                "cnl": "WarfightingFunction subclass-of MilitaryProcess",
                "kif": "(subclass WarfightingFunction MilitaryProcess)",
                "pattern": "subclass",
            }
        ]
        generator_batches = [("instance", [_pair("instance", i) for i in range(3)])]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=3,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
                pinned_pairs=pinned,
            )
            output_cnls = {p["cnl"] for p in result.pairs}
            assert "WarfightingFunction subclass-of MilitaryProcess" in output_cnls
            assert result.pinned_count == 1
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_pinned_pairs_with_benchmark_cnl_are_excluded(self):
        """Pinned pairs whose CNL matches the benchmark are still excluded."""
        pinned_with_benchmark_cnl = [
            {
                "nl": "A different sentence about intelligence processes.",
                "cnl": self._BENCHMARK_REVIEW_CNL,
                "kif": "(subclass IntelligenceProcess MilitaryProcess)",
                "pattern": "subclass",
            }
        ]
        generator_batches = [("instance", [_pair("instance", i) for i in range(2)])]

        scratch_dir = _make_scratch_dir()
        try:
            result = prepare_training_data(
                limit=2,
                output_path=scratch_dir / "train.jsonl",
                generator_batches=generator_batches,
                pinned_pairs=pinned_with_benchmark_cnl,
                exclude_benchmark_overlap=True,
            )
            output_cnls = {p["cnl"] for p in result.pairs}
            assert self._BENCHMARK_REVIEW_CNL not in output_cnls
            assert result.pinned_count == 0
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_real_doctrine_train_file_exists(self):
        assert _REAL_DOCTRINE_TRAIN_PATH.exists(), (
            f"Missing: {_REAL_DOCTRINE_TRAIN_PATH}"
        )

    def test_real_doctrine_pairs_load_and_have_required_keys(self):
        pairs = load_real_doctrine_pairs()
        assert len(pairs) > 0
        for pair in pairs:
            for key in ("nl", "cnl", "kif", "pattern"):
                assert key in pair, f"Missing key '{key}' in pair: {pair}"

    def test_real_doctrine_pairs_all_compile(self):
        compiler = CNLCompiler()
        pairs = load_real_doctrine_pairs()
        for pair in pairs:
            compiled = compiler.compile(pair["cnl"])
            assert compiled == pair["kif"], (
                f"KIF mismatch for CNL '{pair['cnl']}': "
                f"expected {pair['kif']!r}, got {compiled!r}"
            )

    def test_real_doctrine_pairs_not_in_benchmark(self):
        bm_nls, bm_cnls = load_benchmark_overlap_sets()
        pairs = load_real_doctrine_pairs()
        for pair in pairs:
            assert pair["nl"] not in bm_nls, (
                f"Real-doctrine NL is a benchmark NL: {pair['nl']!r}"
            )
            assert pair["cnl"] not in bm_cnls, (
                f"Real-doctrine CNL is a benchmark CNL: {pair['cnl']!r}"
            )
