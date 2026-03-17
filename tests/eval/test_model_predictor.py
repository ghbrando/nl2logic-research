from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from src.eval.evaluate import GoldPair
from src.eval.model_predictor import ModelPredictor


def _make_scratch_dir() -> Path:
    path = Path(".pytest_tmp_eval") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class FakeTensorBatch:
    def __init__(self, values):
        self.values = values
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.prompts_seen: list[str] = []

    def __call__(self, texts=None, **kwargs):
        assert texts is not None
        self.prompts_seen.extend(texts)
        assert kwargs["return_tensors"] == "pt"
        return {
            "input_ids": FakeTensorBatch([[101, 102]]),
            "attention_mask": FakeTensorBatch([[1, 1]]),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        return list(sequences)


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.calls: list[dict] = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ["?x is-a Process"]


class FakeSampler:
    def __init__(self, model, tokenizer, output="?x is-a Process"):
        self.model = model
        self.tokenizer = tokenizer
        self.output = output
        self.calls: list[tuple[str, int]] = []

    def sample(self, prompt: str, max_tokens: int = 200) -> str:
        self.calls.append((prompt, max_tokens))
        return self.output


class ImportErrorSampler:
    def sample(self, prompt: str, max_tokens: int = 200) -> str:
        raise ImportError("outlines is not installed")


class TestModelPredictor:
    def test_returns_predicted_cnl_using_training_prompt_format_in_raw_mode(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        predictor = ModelPredictor(model=model, tokenizer=tokenizer, max_new_tokens=17, decoding="raw")
        pair = GoldPair(
            nl="Something is an instance of Process.",
            cnl="?x is-a Process",
            kif="(instance ?x Process)",
            pattern="instance",
        )

        predicted = predictor(pair)

        assert predicted == "?x is-a Process"
        assert tokenizer.prompts_seen == ["translate to CNL: Something is an instance of Process."]
        assert model.calls[0]["max_new_tokens"] == 17
        assert model.calls[0]["input_ids"].moved_to == "cpu"

    def test_auto_mode_prefers_constrained_sampler(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        sampler = FakeSampler(model, tokenizer, output="Process subclass-of Entity")
        predictor = ModelPredictor(
            model=model,
            tokenizer=tokenizer,
            decoding="auto",
            max_new_tokens=21,
            sampler_factory=lambda _model, _tokenizer: sampler,
        )
        pair = GoldPair(
            nl="Process is a type of Entity.",
            cnl="Process subclass-of Entity",
            kif="(subclass Process Entity)",
            pattern="subclass",
        )

        predicted = predictor(pair)

        assert predicted == "Process subclass-of Entity"
        assert sampler.calls == [("translate to CNL: Process is a type of Entity.", 21)]
        assert model.calls == []
        assert tokenizer.prompts_seen == []

    def test_auto_mode_falls_back_to_raw_when_constrained_unavailable(self, caplog):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        predictor = ModelPredictor(
            model=model,
            tokenizer=tokenizer,
            decoding="auto",
            sampler_factory=lambda _model, _tokenizer: ImportErrorSampler(),
        )
        pair = GoldPair(
            nl="Something is an instance of Process.",
            cnl="?x is-a Process",
            kif="(instance ?x Process)",
            pattern="instance",
        )

        with caplog.at_level("WARNING"):
            predicted = predictor(pair)

        assert predicted == "?x is-a Process"
        assert model.calls[0]["max_new_tokens"] == 64
        assert "falling back to raw generation" in caplog.text

    def test_constrained_mode_raises_when_sampler_is_unavailable(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        predictor = ModelPredictor(
            model=model,
            tokenizer=tokenizer,
            decoding="constrained",
            sampler_factory=lambda _model, _tokenizer: ImportErrorSampler(),
        )
        pair = GoldPair(
            nl="Something is an instance of Process.",
            cnl="?x is-a Process",
            kif="(instance ?x Process)",
            pattern="instance",
        )

        with pytest.raises(ImportError, match="Constrained decoding is unavailable"):
            predictor(pair)

    def test_returns_none_when_input_triggers_abstain(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        predictor = ModelPredictor(model=model, tokenizer=tokenizer, decoding="auto")
        pair = GoldPair(
            nl="If it advances, the unit attacks.",
            cnl="",
            kif="",
            pattern="conditional",
        )

        predicted = predictor(pair)

        assert predicted is None
        assert model.calls == []
        assert tokenizer.prompts_seen == []

    def test_loads_checkpoint_directory_via_loader(self):
        scratch_dir = _make_scratch_dir()
        try:
            loaded_model = FakeModel()
            loaded_tokenizer = FakeTokenizer()
            seen_paths: list[Path] = []

            def loader(path: Path):
                seen_paths.append(path)
                return loaded_model, loaded_tokenizer

            predictor = ModelPredictor(model_path=scratch_dir, model_loader=loader, decoding="raw")
            pair = GoldPair(
                nl="Weapon is a type of Artifact.",
                cnl="Weapon subclass-of Artifact",
                kif="(subclass Weapon Artifact)",
                pattern="subclass",
            )

            predicted = predictor(pair)

            assert seen_paths == [scratch_dir]
            assert predicted == "?x is-a Process"
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_invalid_decoding_mode_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unknown decoding mode"):
            ModelPredictor(model=FakeModel(), tokenizer=FakeTokenizer(), decoding="beam")
