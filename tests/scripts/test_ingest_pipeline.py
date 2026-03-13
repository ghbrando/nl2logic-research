from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import pytest

import scripts.ingest_pipeline as ingest_pipeline
from scripts.ingest_pipeline import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REJECT_PATH,
    KifRecord,
    PipelineConfig,
    PipelineError,
    RejectRecord,
    build_pipeline,
    load_normalizer,
    process_sentence,
    run_pipeline,
)
from src.ingest.extractor import DocSentence
from src.preprocessing.normalize import NormalizationResult


def _make_scratch_dir() -> Path:
    path = Path(".pytest_tmp_scripts") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class _FakeLinker:
    def __init__(self, *, linked_prefix: str = "") -> None:
        self.linked_prefix = linked_prefix

    def link(self, nl: str) -> str:
        return f"{self.linked_prefix}{nl}"


class _FakeSampler:
    last_pattern = "instance"

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    @classmethod
    def abstain_if_unsupported(cls, _nl: str) -> bool:
        return False

    def sample(self, prompt: str) -> str:
        return f"CNL::{prompt}"


class _AbstainingSampler:
    @classmethod
    def abstain_if_unsupported(cls, _nl: str) -> bool:
        return True


class _FakeCompiler:
    def compile(self, cnl: str) -> str:
        return f"KIF::{cnl}"


class TestBuildPipeline:
    def test_default_output_and_reject_paths_are_set_correctly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        args = Namespace(
            pdf=Path("fm2-0.pdf"),
            output=DEFAULT_OUTPUT_PATH,
            rejects=DEFAULT_REJECT_PATH,
            model=Path("models/flan-t5-small-cnl"),
            normalizer="llm",
            anthropic_api_key=None,
            context_window=3,
            verbose=False,
        )

        config = build_pipeline(args)

        assert config.output_path == DEFAULT_OUTPUT_PATH
        assert config.reject_path == DEFAULT_REJECT_PATH

    def test_normalizer_none_sets_field_correctly(self):
        args = Namespace(
            pdf=Path("fm2-0.pdf"),
            output=DEFAULT_OUTPUT_PATH,
            rejects=DEFAULT_REJECT_PATH,
            model=Path("models/flan-t5-small-cnl"),
            normalizer="none",
            anthropic_api_key=None,
            context_window=3,
            verbose=False,
        )

        config = build_pipeline(args)

        assert config.normalizer == "none"

    def test_missing_api_key_with_llm_normalizer_is_caught_in_load_normalizer(self):
        config = PipelineConfig(
            pdf_path=Path("fm2-0.pdf"),
            output_path=DEFAULT_OUTPUT_PATH,
            reject_path=DEFAULT_REJECT_PATH,
            model_path=Path("models/flan-t5-small-cnl"),
            normalizer="llm",
            anthropic_api_key=None,
            context_window=3,
            verbose=False,
        )

        with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY"):
            load_normalizer(config)


class TestProcessSentence:
    def test_successfully_translated_subclaim_produces_kif_record_with_correct_fields(self, monkeypatch):
        doc_sentence = DocSentence("Original sentence.", 5, 2, "TASKS")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Normalized sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )
        monkeypatch.setattr(ingest_pipeline, "decompose", lambda sentence: [sentence])
        monkeypatch.setattr(ingest_pipeline, "CNLSampler", _FakeSampler)

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(linked_prefix="Linked: "),
            _FakeSampler(),
            _FakeCompiler(),
        )

        assert rejects == []
        assert records == [
            KifRecord(
                kif="KIF::CNL::translate to CNL: Linked: Normalized sentence.",
                cnl="CNL::translate to CNL: Linked: Normalized sentence.",
                nl="Linked: Normalized sentence.",
                original="Original sentence.",
                page_number=5,
                position=2,
                section="TASKS",
                pattern="instance",
            )
        ]

    def test_abstained_subclaim_produces_reject_record(self, monkeypatch):
        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Subclaim sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )
        monkeypatch.setattr(ingest_pipeline, "decompose", lambda sentence: [sentence])
        monkeypatch.setattr(ingest_pipeline, "CNLSampler", _AbstainingSampler)

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
        )

        assert records == []
        assert rejects == [
            RejectRecord(
                original="Original sentence.",
                subclaim="Subclaim sentence.",
                page_number=1,
                position=0,
                section="",
                reason="abstained",
                detail="Input appears outside the supported fragment.",
            )
        ]

    def test_compile_error_produces_reject_record(self, monkeypatch):
        class BrokenCompiler:
            def compile(self, _cnl: str) -> str:
                raise ValueError("bad cnl")

        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Subclaim sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )
        monkeypatch.setattr(ingest_pipeline, "decompose", lambda sentence: [sentence])
        monkeypatch.setattr(ingest_pipeline, "CNLSampler", _FakeSampler)

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            BrokenCompiler(),
        )

        assert records == []
        assert rejects[0].reason == "compile_error"
        assert rejects[0].detail == "bad cnl"

    def test_unexpected_exception_produces_exception_reject_record(self, monkeypatch):
        class BrokenLinker:
            def link(self, _nl: str) -> str:
                raise RuntimeError("link blew up")

        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Subclaim sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            BrokenLinker(),
            _FakeSampler(),
            _FakeCompiler(),
        )

        assert records == []
        assert rejects == [
            RejectRecord(
                original="Original sentence.",
                subclaim="Original sentence.",
                page_number=1,
                position=0,
                section="",
                reason="exception",
                detail="link blew up",
            )
        ]

    def test_normalization_ambiguous_sentence_produces_reject_record(self, monkeypatch):
        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=[],
                ambiguous=["Ambiguous sentence."],
                original=sentence,
            ),
        )

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
        )

        assert records == []
        assert rejects == [
            RejectRecord(
                original="Original sentence.",
                subclaim="Ambiguous sentence.",
                page_number=1,
                position=0,
                section="",
                reason="normalization_ambiguous",
                detail="Normalization returned an ambiguous line.",
            )
        ]

    def test_multiple_subclaims_produce_multiple_records(self, monkeypatch):
        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Normalized sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )
        monkeypatch.setattr(
            ingest_pipeline,
            "decompose",
            lambda sentence: [f"{sentence} one", f"{sentence} two"],
        )
        monkeypatch.setattr(ingest_pipeline, "CNLSampler", _FakeSampler)

        records, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
        )

        assert len(records) == 2
        assert rejects == []


class TestRunPipeline:
    def test_records_are_written_incrementally_and_stats_are_correct(self, monkeypatch):
        scratch_dir = _make_scratch_dir()
        try:
            pdf_path = scratch_dir / "fm2-0.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")
            model_path = scratch_dir / "model"
            model_path.mkdir()
            output_path = scratch_dir / "ontology.jsonl"
            reject_path = scratch_dir / "rejects.jsonl"

            doc_sentences = [
                DocSentence("Sentence one.", 1, 0, "TASKS"),
                DocSentence("Sentence two.", 1, 1, "TASKS"),
            ]
            normalize_calls: list[tuple[str, list[str]]] = []

            monkeypatch.setattr(ingest_pipeline, "extract_sentences", lambda path: doc_sentences)
            monkeypatch.setattr(ingest_pipeline, "load_model_and_tokenizer", lambda path: ("model", "tokenizer"))
            monkeypatch.setattr(ingest_pipeline, "EntityLinker", lambda: _FakeLinker())
            monkeypatch.setattr(ingest_pipeline, "load_normalizer", lambda config: (lambda sentence, context: f"NORM::{sentence}"))
            monkeypatch.setattr(
                ingest_pipeline,
                "normalize",
                lambda sentence, context, provider: (
                    normalize_calls.append((sentence, list(context)))
                    or NormalizationResult(
                        normalized=[f"NORM::{sentence}"],
                        ambiguous=[],
                        original=sentence,
                    )
                ),
            )
            monkeypatch.setattr(
                ingest_pipeline,
                "decompose",
                lambda sentence: [sentence] if "one" in sentence else [sentence, f"{sentence} extra"],
            )

            class FakeSampler:
                last_pattern = "instance"

                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                @classmethod
                def abstain_if_unsupported(cls, nl: str) -> bool:
                    return nl.endswith(" extra")

                def sample(self, prompt: str) -> str:
                    return f"CNL::{prompt}"

            monkeypatch.setattr(ingest_pipeline, "CNLSampler", FakeSampler)
            monkeypatch.setattr(ingest_pipeline, "CNLCompiler", lambda: _FakeCompiler())

            write_calls: list[tuple[str, int]] = []
            original_write_records = ingest_pipeline._write_records

            def recording_write_records(handle, records):
                write_calls.append((Path(handle.name).name, len(records)))
                return original_write_records(handle, records)

            monkeypatch.setattr(ingest_pipeline, "_write_records", recording_write_records)

            config = PipelineConfig(
                pdf_path=pdf_path,
                output_path=output_path,
                reject_path=reject_path,
                model_path=model_path,
                normalizer="nlp",
                anthropic_api_key=None,
                context_window=1,
                verbose=False,
            )

            stats = run_pipeline(config)

            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            reject_rows = [json.loads(line) for line in reject_path.read_text(encoding="utf-8").splitlines()]

            assert len(output_rows) == 2
            assert len(reject_rows) == 1
            assert write_calls == [("ontology.jsonl", 1), ("ontology.jsonl", 1), ("rejects.jsonl", 1)]
            assert stats.total_sentences == 2
            assert stats.total_subclaims == 3
            assert stats.translated == 2
            assert stats.abstained == 1
            assert stats.coverage == pytest.approx(2 / 3)
            assert normalize_calls == [
                ("Sentence one.", []),
                ("Sentence two.", ["NORM::Sentence one."]),
            ]
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def test_normalizer_none_skips_normalization_stage(self, monkeypatch):
        scratch_dir = _make_scratch_dir()
        try:
            pdf_path = scratch_dir / "fm2-0.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")
            model_path = scratch_dir / "model"
            model_path.mkdir()
            output_path = scratch_dir / "ontology.jsonl"
            reject_path = scratch_dir / "rejects.jsonl"

            monkeypatch.setattr(
                ingest_pipeline,
                "extract_sentences",
                lambda path: [DocSentence("Sentence one.", 1, 0, "")],
            )
            monkeypatch.setattr(ingest_pipeline, "load_model_and_tokenizer", lambda path: ("model", "tokenizer"))
            monkeypatch.setattr(ingest_pipeline, "EntityLinker", lambda: _FakeLinker())
            monkeypatch.setattr(ingest_pipeline, "load_normalizer", lambda config: None)

            normalize_called = False

            def fake_normalize(sentence, context, provider):
                nonlocal normalize_called
                normalize_called = True
                return NormalizationResult(normalized=[sentence], ambiguous=[], original=sentence)

            monkeypatch.setattr(ingest_pipeline, "normalize", fake_normalize)
            monkeypatch.setattr(ingest_pipeline, "decompose", lambda sentence: [sentence])
            monkeypatch.setattr(ingest_pipeline, "CNLSampler", _FakeSampler)
            monkeypatch.setattr(ingest_pipeline, "CNLCompiler", lambda: _FakeCompiler())

            stats = run_pipeline(
                PipelineConfig(
                    pdf_path=pdf_path,
                    output_path=output_path,
                    reject_path=reject_path,
                    model_path=model_path,
                    normalizer="none",
                    anthropic_api_key=None,
                    context_window=3,
                    verbose=False,
                )
            )

            assert normalize_called is False
            assert stats.translated == 1
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
