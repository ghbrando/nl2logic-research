from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from src.training.train import (
    TrainingPair,
    build_pattern_coverage_pairs,
    build_tokenized_records,
    collect_sanity_check_rows,
    collect_pattern_coverage_rows,
    format_prompt,
    load_training_pairs,
    split_training_pairs,
)


class FakeTokenizer:
    def __init__(self):
        self.prompts_seen: list[str] = []
        self.targets_seen: list[str] = []

    def __call__(self, texts=None, **kwargs):
        if "text_target" in kwargs:
            targets = kwargs["text_target"]
            self.targets_seen.extend(targets)
            return {"input_ids": [[301, 302] for _ in targets]}

        assert texts is not None
        self.prompts_seen.extend(texts)
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": FakeTensorBatch(
                    [[10 * (index + 1) + 1, 10 * (index + 1) + 2] for index, _ in enumerate(texts)]
                ),
                "attention_mask": FakeTensorBatch([[1, 1] for _ in texts]),
            }

        return {
            "input_ids": [[101, 102] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }

    def as_target_tokenizer(self):
        return nullcontext(self)

    def batch_decode(self, generated, skip_special_tokens=True):
        assert skip_special_tokens is True
        return list(generated)


class FakeTensorBatch:
    def __init__(self, values):
        self.values = values
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeModel:
    def __init__(self):
        self.device = "cuda:0"
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"].values
        return [f"?x is-a Process #{index}" for index, _ in enumerate(input_ids, start=1)]


class FakeCompiler:
    def compile(self, cnl: str) -> str:
        if "Bad" in cnl:
            raise ValueError("bad cnl")
        return f"(compiled {cnl})"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class TestLoadTrainingPairs:
    def test_loads_jsonl_records_and_limit(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        _write_jsonl(
            path,
            [
                {"nl": "One", "cnl": "?x is-a Process", "kif": "(instance ?x Process)"},
                {"nl": "Two", "cnl": "Process subclass-of Entity"},
            ],
        )

        pairs = load_training_pairs(path, limit=1)

        assert pairs == [
            TrainingPair(
                nl="One",
                cnl="?x is-a Process",
                kif="(instance ?x Process)",
                pattern=None,
            )
        ]

    def test_raises_for_missing_required_fields(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        _write_jsonl(path, [{"nl": "Missing cnl"}])

        with pytest.raises(ValueError, match="'nl' and 'cnl'"):
            load_training_pairs(path, limit=0)


class TestFormatAndSplit:
    def test_formats_prompt(self):
        assert format_prompt("Every unit advances.") == "translate to CNL: Every unit advances."

    def test_split_is_deterministic_and_non_empty(self):
        pairs = [
            TrainingPair(nl=f"nl-{index}", cnl=f"cnl-{index}")
            for index in range(10)
        ]

        train_a, val_a = split_training_pairs(pairs, seed=7)
        train_b, val_b = split_training_pairs(pairs, seed=7)

        assert train_a == train_b
        assert val_a == val_b
        assert len(train_a) == 9
        assert len(val_a) == 1


class TestTokenization:
    def test_build_tokenized_records_uses_prompt_prefix_and_labels(self):
        tokenizer = FakeTokenizer()
        pairs = [TrainingPair(nl="Advance to contact.", cnl="?x is-a Process")]

        records = build_tokenized_records(pairs, tokenizer)

        assert tokenizer.prompts_seen == ["translate to CNL: Advance to contact."]
        assert tokenizer.targets_seen == ["?x is-a Process"]
        assert records == [
            {
                "input_ids": [101, 102],
                "attention_mask": [1, 1],
                "labels": [301, 302],
            }
        ]


class TestSanityCheck:
    def test_collects_nl_cnl_kif_rows_with_mock_model(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        held_out = [
            TrainingPair(nl="A process exists.", cnl="some ?x is-a Process"),
            TrainingPair(nl="Every process is an entity.", cnl="Process subclass-of Entity"),
        ]

        rows = collect_sanity_check_rows(
            model,
            tokenizer,
            held_out,
            compiler=FakeCompiler(),
            sample_size=2,
        )

        assert len(rows) == 2
        assert rows[0]["nl"] == "A process exists."
        assert rows[0]["cnl"] == "?x is-a Process #1"
        assert rows[0]["kif"] == "(compiled ?x is-a Process #1)"
        assert rows[1]["cnl"] == "?x is-a Process #2"
        assert model.calls[0]["max_new_tokens"] == 64
        assert model.calls[0]["input_ids"].moved_to == "cuda:0"


class TestPatternCoverage:
    def test_build_pattern_coverage_pairs_skips_nary_when_absent(self):
        pairs = [
            TrainingPair(nl="nl", cnl="cnl", pattern="instance"),
            TrainingPair(nl="nl", cnl="cnl", pattern="subclass"),
            TrainingPair(nl="nl", cnl="cnl", pattern="binary"),
            TrainingPair(nl="nl", cnl="cnl", pattern="conditional"),
            TrainingPair(nl="nl", cnl="cnl", pattern="existential"),
        ]

        probes, skipped = build_pattern_coverage_pairs(pairs)

        assert [probe.pattern for probe in probes] == [
            "instance",
            "subclass",
            "binary",
            "conditional",
            "existential",
        ]
        assert skipped == ["nary"]

    def test_collect_pattern_coverage_rows_preserves_pattern_labels(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        pairs = [
            TrainingPair(nl="nl", cnl="cnl", pattern="instance"),
            TrainingPair(nl="nl", cnl="cnl", pattern="subclass"),
            TrainingPair(nl="nl", cnl="cnl", pattern="binary"),
            TrainingPair(nl="nl", cnl="cnl", pattern="conditional"),
            TrainingPair(nl="nl", cnl="cnl", pattern="existential"),
            TrainingPair(nl="nl", cnl="cnl", pattern="nary"),
        ]

        rows, skipped = collect_pattern_coverage_rows(
            model,
            tokenizer,
            pairs,
            compiler=FakeCompiler(),
        )

        assert skipped == []
        assert [row["pattern"] for row in rows] == [
            "instance",
            "subclass",
            "binary",
            "conditional",
            "existential",
            "nary",
        ]
        assert rows[0]["nl"] == "Something is an instance of Process."
        assert rows[-1]["nl"] == "The between relation holds among three entities."
