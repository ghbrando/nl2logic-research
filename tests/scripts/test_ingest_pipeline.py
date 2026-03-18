from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import pytest

import scripts.ingest_pipeline as ingest_pipeline
from scripts.ingest_pipeline import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REJECT_PATH,
    DEFAULT_REVIEW_PATH,
    KifRecord,
    PipelineConfig,
    PipelineError,
    RejectRecord,
    ReviewRecord,
    build_pipeline,
    build_query_index,
    load_normalizer,
    process_sentence,
    run_pipeline,
)
from src.ingest.extractor import DocSentence
from src.ingest.grounding import GroundingAssessment
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


class _AcceptingGrounder:
    def assess(self, **_kwargs) -> GroundingAssessment:
        return GroundingAssessment(
            accepted=True,
            reason=None,
            detail="grounded",
            relation_terms=["agent"],
            class_terms=["MilitaryProcess", "AutonomousAgent"],
            terms=["agent", "MilitaryProcess", "AutonomousAgent"],
            ungrounded_terms=[],
        )


class _ReviewingGrounder:
    def assess(self, **_kwargs) -> GroundingAssessment:
        return GroundingAssessment(
            accepted=False,
            reason="ungrounded_class_terms",
            detail="Generated class terms are not lexically supported by the source text: AutonomousAgent",
            relation_terms=["agent"],
            class_terms=["MilitaryProcess", "AutonomousAgent"],
            terms=["agent", "MilitaryProcess", "AutonomousAgent"],
            ungrounded_terms=["AutonomousAgent"],
        )


class TestBuildPipeline:
    def test_default_output_review_reject_and_index_paths_are_set_correctly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        args = Namespace(
            pdf=Path("fm2-0.pdf"),
            output=DEFAULT_OUTPUT_PATH,
            review=DEFAULT_REVIEW_PATH,
            rejects=DEFAULT_REJECT_PATH,
            index=DEFAULT_INDEX_PATH,
            model=Path("models/flan-t5-small-cnl"),
            normalizer="llm",
            anthropic_api_key=None,
            context_window=3,
            verbose=False,
        )

        config = build_pipeline(args)

        assert config.output_path == DEFAULT_OUTPUT_PATH
        assert config.review_path == DEFAULT_REVIEW_PATH
        assert config.reject_path == DEFAULT_REJECT_PATH
        assert config.index_path == DEFAULT_INDEX_PATH

    def test_normalizer_none_sets_field_correctly(self):
        args = Namespace(
            pdf=Path("fm2-0.pdf"),
            output=DEFAULT_OUTPUT_PATH,
            review=DEFAULT_REVIEW_PATH,
            rejects=DEFAULT_REJECT_PATH,
            index=DEFAULT_INDEX_PATH,
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
            review_path=DEFAULT_REVIEW_PATH,
            reject_path=DEFAULT_REJECT_PATH,
            index_path=DEFAULT_INDEX_PATH,
            model_path=Path("models/flan-t5-small-cnl"),
            normalizer="llm",
            anthropic_api_key=None,
            context_window=3,
            verbose=False,
        )

        with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY"):
            load_normalizer(config)


class TestProcessSentence:
    def test_successfully_translated_subclaim_produces_accepted_kif_record(self, monkeypatch):
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(linked_prefix="Linked: "),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert reviews == []
        assert rejects == []
        assert records == [
            KifRecord(
                record_id="p5-pos2-n0",
                kif="KIF::CNL::translate to CNL: Linked: Normalized sentence.",
                cnl="CNL::translate to CNL: Linked: Normalized sentence.",
                nl="Linked: Normalized sentence.",
                original="Original sentence.",
                page_number=5,
                position=2,
                section="TASKS",
                pattern="instance",
                relation="agent",
                terms=["agent", "MilitaryProcess", "AutonomousAgent"],
            )
        ]

    def test_grounding_failure_routes_record_to_review_queue(self, monkeypatch):
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_ReviewingGrounder(),
        )

        assert records == []
        assert rejects == []
        assert reviews == [
            ReviewRecord(
                record_id="p1-pos0-n0",
                cnl="CNL::translate to CNL: Subclaim sentence.",
                kif="KIF::CNL::translate to CNL: Subclaim sentence.",
                subclaim="Subclaim sentence.",
                original="Original sentence.",
                page_number=1,
                position=0,
                section="",
                pattern="instance",
                relation="agent",
                terms=["agent", "MilitaryProcess", "AutonomousAgent"],
                reason="ungrounded_class_terms",
                detail="Generated class terms are not lexically supported by the source text: AutonomousAgent",
                ungrounded_terms=["AutonomousAgent"],
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert records == []
        assert reviews == []
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

    def test_noise_subclaim_is_rejected_before_model_generation(self, monkeypatch):
        doc_sentence = DocSentence("Original sentence.", 1, 0, "")
        monkeypatch.setattr(
            ingest_pipeline,
            "normalize",
            lambda sentence, context, provider: NormalizationResult(
                normalized=["Original sentence."],
                ambiguous=[],
                original=sentence,
            ),
        )
        monkeypatch.setattr(ingest_pipeline, "decompose", lambda sentence: ["⚫ Bullet fragment without proposition"])
        monkeypatch.setattr(ingest_pipeline, "CNLSampler", _FakeSampler)

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert records == []
        assert reviews == []
        assert rejects == [
            RejectRecord(
                original="Original sentence.",
                subclaim="⚫ Bullet fragment without proposition",
                page_number=1,
                position=0,
                section="",
                reason="noise",
                detail="Subclaim looks like doctrine formatting noise or citation text.",
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            BrokenCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert records == []
        assert reviews == []
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            BrokenLinker(),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert records == []
        assert reviews == []
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

        records, reviews, rejects = process_sentence(
            doc_sentence,
            [],
            lambda sentence, context: sentence,
            _FakeLinker(),
            _FakeSampler(),
            _FakeCompiler(),
            grounder=_AcceptingGrounder(),
        )

        assert records == []
        assert reviews == []
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


class TestBuildQueryIndex:
    def test_records_are_indexed_by_relation_term_section_and_page(self):
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
            )
        ]

        index = build_query_index(records)

        assert index["stats"]["record_count"] == 1
        assert index["by_relation"] == {"agent": ["p1-pos0-n0"]}
        assert index["by_term"]["MilitaryProcess"] == ["p1-pos0-n0"]
        assert index["by_section"] == {"TASKS": ["p1-pos0-n0"]}
        assert index["by_page"] == {"1": ["p1-pos0-n0"]}


class TestRunPipeline:
    def test_records_reviews_rejects_and_index_are_written_and_stats_are_correct(self, monkeypatch):
        scratch_dir = _make_scratch_dir()
        try:
            pdf_path = scratch_dir / "fm2-0.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")
            model_path = scratch_dir / "model"
            model_path.mkdir()
            output_path = scratch_dir / "ontology.jsonl"
            review_path = scratch_dir / "review.jsonl"
            reject_path = scratch_dir / "rejects.jsonl"
            index_path = scratch_dir / "index.json"

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

            class FakeGrounder:
                def assess(self, *, original_text: str, **_kwargs) -> GroundingAssessment:
                    if original_text == "Sentence two.":
                        return GroundingAssessment(
                            accepted=False,
                            reason="ungrounded_class_terms",
                            detail="Generated class terms are not lexically supported by the source text: AutonomousAgent",
                            relation_terms=["agent"],
                            class_terms=["MilitaryProcess", "AutonomousAgent"],
                            terms=["agent", "MilitaryProcess", "AutonomousAgent"],
                            ungrounded_terms=["AutonomousAgent"],
                        )
                    return GroundingAssessment(
                        accepted=True,
                        reason=None,
                        detail="grounded",
                        relation_terms=["agent"],
                        class_terms=["MilitaryProcess", "AutonomousAgent"],
                        terms=["agent", "MilitaryProcess", "AutonomousAgent"],
                        ungrounded_terms=[],
                    )

            monkeypatch.setattr(ingest_pipeline, "CNLSampler", FakeSampler)
            monkeypatch.setattr(ingest_pipeline, "CNLCompiler", lambda: _FakeCompiler())
            monkeypatch.setattr(ingest_pipeline, "DoctrineGrounder", lambda: FakeGrounder())

            write_calls: list[tuple[str, int]] = []
            original_write_records = ingest_pipeline._write_records

            def recording_write_records(handle, records):
                write_calls.append((Path(handle.name).name, len(records)))
                return original_write_records(handle, records)

            monkeypatch.setattr(ingest_pipeline, "_write_records", recording_write_records)

            config = PipelineConfig(
                pdf_path=pdf_path,
                output_path=output_path,
                review_path=review_path,
                reject_path=reject_path,
                index_path=index_path,
                model_path=model_path,
                normalizer="nlp",
                anthropic_api_key=None,
                context_window=1,
                verbose=False,
            )

            stats = run_pipeline(config)

            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
            reject_rows = [json.loads(line) for line in reject_path.read_text(encoding="utf-8").splitlines()]
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))

            assert len(output_rows) == 1
            assert len(review_rows) == 1
            assert len(reject_rows) == 1
            assert write_calls == [
                ("ontology.jsonl", 1),
                ("review.jsonl", 1),
                ("rejects.jsonl", 1),
            ]
            assert index_payload["stats"]["record_count"] == 1
            assert index_payload["by_relation"] == {"agent": ["p1-pos0-n0"]}
            assert stats.total_sentences == 2
            assert stats.total_subclaims == 3
            assert stats.translated == 1
            assert stats.reviewed == 1
            assert stats.abstained == 1
            assert stats.coverage == pytest.approx(1 / 3)
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
            review_path = scratch_dir / "review.jsonl"
            reject_path = scratch_dir / "rejects.jsonl"
            index_path = scratch_dir / "index.json"

            monkeypatch.setattr(
                ingest_pipeline,
                "extract_sentences",
                lambda path: [DocSentence("Sentence one.", 1, 0, "")],
            )
            monkeypatch.setattr(ingest_pipeline, "load_model_and_tokenizer", lambda path: ("model", "tokenizer"))
            monkeypatch.setattr(ingest_pipeline, "EntityLinker", lambda: _FakeLinker())
            monkeypatch.setattr(ingest_pipeline, "load_normalizer", lambda config: None)
            monkeypatch.setattr(ingest_pipeline, "DoctrineGrounder", lambda: _AcceptingGrounder())

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
                    review_path=review_path,
                    reject_path=reject_path,
                    index_path=index_path,
                    model_path=model_path,
                    normalizer="none",
                    anthropic_api_key=None,
                    context_window=3,
                    verbose=False,
                )
            )

            assert normalize_called is False
            assert stats.translated == 1
            assert stats.reviewed == 0
            assert json.loads(index_path.read_text(encoding="utf-8"))["stats"]["record_count"] == 1
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
