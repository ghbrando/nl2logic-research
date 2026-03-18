from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.compiler.compiler import CNLCompiler
from src.eval.model_predictor import DEFAULT_MODEL_PATH, load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.ingest.extractor import DocSentence, ExtractionError, extract_sentences
from src.ingest.grounding import DoctrineGrounder
from src.preprocessing.decompose import decompose
from src.preprocessing.entity_linker import EntityLinker
from src.preprocessing.normalize import (
    AnthropicAdapter,
    NLPAdapter,
    NormalizationResult,
    normalize,
)
from src.training.train import format_prompt

DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_kif.jsonl"
DEFAULT_REVIEW_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_review.jsonl"
DEFAULT_REJECT_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_rejects.jsonl"
DEFAULT_INDEX_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_index.json"
DEFAULT_CONTEXT_WINDOW = 3

_BULLET_SYMBOL_RE = re.compile(r"[▪⚫•◦●■]")
_NOISE_PATTERNS = (
    re.compile(r"\bFM\s+\d[\d-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:ATP|ADP|ADRP|JP)\s+\d[\d.-]*\b", re.IGNORECASE),
    re.compile(r"^\(see\b", re.IGNORECASE),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\d+-\d+\b"),
)


class PipelineError(Exception):
    """Raised when the ingest pipeline cannot be initialised safely."""


@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path
    output_path: Path
    review_path: Path
    reject_path: Path
    index_path: Path
    model_path: Path
    normalizer: str
    anthropic_api_key: str | None
    context_window: int
    verbose: bool


@dataclass(frozen=True)
class KifRecord:
    record_id: str
    kif: str
    cnl: str
    nl: str
    original: str
    page_number: int
    position: int
    section: str
    pattern: str
    relation: str | None
    terms: list[str]


@dataclass(frozen=True)
class ReviewRecord:
    record_id: str
    cnl: str
    kif: str | None
    subclaim: str
    original: str
    page_number: int
    position: int
    section: str
    pattern: str
    relation: str | None
    terms: list[str]
    reason: str
    detail: str
    ungrounded_terms: list[str]


@dataclass(frozen=True)
class RejectRecord:
    original: str
    subclaim: str
    page_number: int
    position: int
    section: str
    reason: str
    detail: str


@dataclass
class PipelineStats:
    total_sentences: int = 0
    total_subclaims: int = 0
    translated: int = 0
    reviewed: int = 0
    abstained: int = 0
    noise_filtered: int = 0
    normalization_ambiguous: int = 0
    compile_errors: int = 0
    model_errors: int = 0
    exceptions: int = 0

    @property
    def coverage(self) -> float:
        return self.translated / self.total_subclaims if self.total_subclaims else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end NL2Logic ingest pipeline over a doctrine PDF."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"KIF JSONL output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW_PATH,
        help=f"Review-log JSONL output path (default: {DEFAULT_REVIEW_PATH})",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=DEFAULT_REJECT_PATH,
        help=f"Reject-log JSONL output path (default: {DEFAULT_REJECT_PATH})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Accepted-facts query index output path (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Seq2seq checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--normalizer",
        choices=("llm", "nlp", "none"),
        default="llm",
        help="Normalization backend (default: llm)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY when omitted.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"Number of prior normalized sentences to pass as context (default: {DEFAULT_CONTEXT_WINDOW})",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit per-sentence warnings during processing.")
    return parser.parse_args(argv)


def build_pipeline(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        pdf_path=Path(args.pdf),
        output_path=Path(args.output),
        review_path=Path(args.review),
        reject_path=Path(args.rejects),
        index_path=Path(args.index),
        model_path=Path(args.model),
        normalizer=args.normalizer,
        anthropic_api_key=args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        context_window=max(0, int(args.context_window)),
        verbose=bool(args.verbose),
    )


def load_normalizer(config: PipelineConfig) -> Callable[[str, list[str]], str] | None:
    if config.normalizer == "none":
        return None
    if config.normalizer == "nlp":
        return NLPAdapter().complete
    if config.normalizer == "llm":
        if not config.anthropic_api_key:
            raise PipelineError("ANTHROPIC_API_KEY is required when --normalizer llm is selected.")
        return AnthropicAdapter(api_key=config.anthropic_api_key).complete
    raise PipelineError(f"Unsupported normalizer backend: {config.normalizer}")


def _default_normalization_result(
    doc_sentence: DocSentence,
    context: list[str],
    normalizer: Callable[[str, list[str]], str] | None,
) -> NormalizationResult:
    if normalizer is None:
        return NormalizationResult(
            normalized=[doc_sentence.sentence],
            ambiguous=[],
            original=doc_sentence.sentence,
        )
    return normalize(doc_sentence.sentence, context, normalizer)


def _prepare_normalization(
    doc_sentence: DocSentence,
    context: list[str],
    normalizer: Callable[[str, list[str]], str] | None,
) -> tuple[NormalizationResult | None, RejectRecord | None]:
    try:
        return _default_normalization_result(doc_sentence, context, normalizer), None
    except Exception as exc:
        return None, _reject(
            doc_sentence,
            subclaim=doc_sentence.sentence,
            reason="exception",
            detail=str(exc),
        )


def _reject(
    doc_sentence: DocSentence,
    *,
    subclaim: str,
    reason: str,
    detail: str,
) -> RejectRecord:
    return RejectRecord(
        original=doc_sentence.sentence,
        subclaim=subclaim,
        page_number=doc_sentence.page_number,
        position=doc_sentence.position,
        section=doc_sentence.section,
        reason=reason,
        detail=detail,
    )


def _get_sampler_pattern(sampler: CNLSampler) -> str:
    for attribute in ("last_pattern", "pattern", "current_pattern"):
        value = getattr(sampler, attribute, "")
        if isinstance(value, str):
            return value
    return ""


def _make_record_id(doc_sentence: DocSentence, ordinal: int) -> str:
    return f"p{doc_sentence.page_number}-pos{doc_sentence.position}-n{ordinal}"


def _looks_like_noise(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return True
    if _BULLET_SYMBOL_RE.search(normalized):
        return True
    if normalized.startswith(("(", "[")) and normalized.lower().startswith(("(see", "[see")):
        return True
    return any(pattern.search(normalized) for pattern in _NOISE_PATTERNS)


def build_query_index(records: list[KifRecord]) -> dict[str, Any]:
    serializable_records = [asdict(record) for record in records]
    by_relation: dict[str, list[str]] = {}
    by_term: dict[str, list[str]] = {}
    by_section: dict[str, list[str]] = {}
    by_page: dict[str, list[str]] = {}

    for record in records:
        record_id = record.record_id
        if record.relation is not None:
            by_relation.setdefault(record.relation, []).append(record_id)
        for term in record.terms:
            by_term.setdefault(term, []).append(record_id)
        by_section.setdefault(record.section, []).append(record_id)
        by_page.setdefault(str(record.page_number), []).append(record_id)

    return {
        "records": serializable_records,
        "by_relation": by_relation,
        "by_term": by_term,
        "by_section": by_section,
        "by_page": by_page,
        "stats": {
            "record_count": len(records),
            "relations": {key: len(value) for key, value in sorted(by_relation.items())},
            "terms": {key: len(value) for key, value in sorted(by_term.items())},
            "sections": {key: len(value) for key, value in sorted(by_section.items())},
            "pages": {key: len(value) for key, value in sorted(by_page.items(), key=lambda item: int(item[0]))},
        },
    }


def _write_records(handle, records: list[Any]) -> None:
    for record in records:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    handle.flush()


def _write_index(path: Path, records: list[KifRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_query_index(records), ensure_ascii=False, indent=2), encoding="utf-8")


def process_sentence(
    doc_sentence: DocSentence,
    context: list[str],
    normalizer: Callable[[str, list[str]], str] | None,
    entity_linker: EntityLinker,
    sampler: CNLSampler,
    compiler: CNLCompiler,
    *,
    normalization_result: NormalizationResult | None = None,
    grounder: DoctrineGrounder | None = None,
) -> tuple[list[KifRecord], list[ReviewRecord], list[RejectRecord]]:
    try:
        result = normalization_result
        if result is None:
            result, normalization_reject = _prepare_normalization(doc_sentence, context, normalizer)
            if normalization_reject is not None or result is None:
                fallback_reject = normalization_reject or _reject(
                    doc_sentence,
                    subclaim=doc_sentence.sentence,
                    reason="exception",
                    detail="Normalization failed unexpectedly.",
                )
                return [], [], [fallback_reject]

        grounder = grounder or DoctrineGrounder()
        kif_records: list[KifRecord] = []
        review_records: list[ReviewRecord] = []
        reject_records: list[RejectRecord] = []
        record_ordinal = 0

        for ambiguous_line in result.ambiguous:
            reject_records.append(
                _reject(
                    doc_sentence,
                    subclaim=ambiguous_line or doc_sentence.sentence,
                    reason="normalization_ambiguous",
                    detail="Normalization returned an ambiguous line.",
                )
            )

        if not result.normalized:
            return kif_records, review_records, reject_records

        for normalized_sentence in result.normalized:
            linked_sentence = entity_linker.link(normalized_sentence)
            subclaims = decompose(linked_sentence)
            if not subclaims:
                reject_records.append(
                    _reject(
                        doc_sentence,
                        subclaim=linked_sentence,
                        reason="noise" if _looks_like_noise(linked_sentence) else "abstained",
                        detail="No translatable subclaims after decomposition.",
                    )
                )
                continue

            for subclaim in subclaims:
                if _looks_like_noise(subclaim):
                    reject_records.append(
                        _reject(
                            doc_sentence,
                            subclaim=subclaim,
                            reason="noise",
                            detail="Subclaim looks like doctrine formatting noise or citation text.",
                        )
                    )
                    continue

                try:
                    if CNLSampler.abstain_if_unsupported(subclaim):
                        reject_records.append(
                            _reject(
                                doc_sentence,
                                subclaim=subclaim,
                                reason="abstained",
                                detail="Input appears outside the supported fragment.",
                            )
                        )
                        continue

                    cnl = sampler.sample(format_prompt(subclaim))
                except UnsupportedInputError as exc:
                    reject_records.append(
                        _reject(
                            doc_sentence,
                            subclaim=subclaim,
                            reason="abstained",
                            detail=str(exc),
                        )
                    )
                    continue
                except Exception as exc:
                    reject_records.append(
                        _reject(
                            doc_sentence,
                            subclaim=subclaim,
                            reason="model_error",
                            detail=str(exc),
                        )
                    )
                    continue

                try:
                    kif = compiler.compile(cnl)
                except Exception as exc:
                    reject_records.append(
                        _reject(
                            doc_sentence,
                            subclaim=subclaim,
                            reason="compile_error",
                            detail=str(exc),
                        )
                    )
                    continue

                record_id = _make_record_id(doc_sentence, record_ordinal)
                record_ordinal += 1
                assessment = grounder.assess(
                    cnl=cnl,
                    original_text=doc_sentence.sentence,
                    normalized_text=normalized_sentence,
                    linked_text=linked_sentence,
                    subclaim_text=subclaim,
                )
                relation = assessment.relation_terms[0] if assessment.relation_terms else None
                pattern = _get_sampler_pattern(sampler)

                if assessment.accepted:
                    kif_records.append(
                        KifRecord(
                            record_id=record_id,
                            kif=kif,
                            cnl=cnl,
                            nl=subclaim,
                            original=doc_sentence.sentence,
                            page_number=doc_sentence.page_number,
                            position=doc_sentence.position,
                            section=doc_sentence.section,
                            pattern=pattern,
                            relation=relation,
                            terms=assessment.terms,
                        )
                    )
                    continue

                review_records.append(
                    ReviewRecord(
                        record_id=record_id,
                        cnl=cnl,
                        kif=kif,
                        subclaim=subclaim,
                        original=doc_sentence.sentence,
                        page_number=doc_sentence.page_number,
                        position=doc_sentence.position,
                        section=doc_sentence.section,
                        pattern=pattern,
                        relation=relation,
                        terms=assessment.terms,
                        reason=assessment.reason or "weak_grounding",
                        detail=assessment.detail,
                        ungrounded_terms=assessment.ungrounded_terms,
                    )
                )

        return kif_records, review_records, reject_records
    except Exception as exc:
        return [], [], [
            _reject(
                doc_sentence,
                subclaim=doc_sentence.sentence,
                reason="exception",
                detail=str(exc),
            )
        ]


def _update_stats(
    stats: PipelineStats,
    kif_records: list[KifRecord],
    review_records: list[ReviewRecord],
    reject_records: list[RejectRecord],
) -> None:
    stats.translated += len(kif_records)
    stats.reviewed += len(review_records)
    stats.total_subclaims += len(kif_records) + len(review_records) + len(reject_records)

    for reject in reject_records:
        if reject.reason == "abstained":
            stats.abstained += 1
        elif reject.reason == "noise":
            stats.noise_filtered += 1
        elif reject.reason == "normalization_ambiguous":
            stats.normalization_ambiguous += 1
        elif reject.reason == "compile_error":
            stats.compile_errors += 1
        elif reject.reason == "model_error":
            stats.model_errors += 1
        elif reject.reason == "exception":
            stats.exceptions += 1


def run_pipeline(config: PipelineConfig) -> PipelineStats:
    if not config.pdf_path.exists():
        raise PipelineError(f"PDF not found: {config.pdf_path}")
    if not config.model_path.exists():
        raise PipelineError(f"Model checkpoint directory not found: {config.model_path}")

    try:
        normalizer = load_normalizer(config)
        model, tokenizer = load_model_and_tokenizer(config.model_path)
        entity_linker = EntityLinker()
        sampler = CNLSampler(model, tokenizer)
        compiler = CNLCompiler()
        grounder = DoctrineGrounder()
        sentences = extract_sentences(config.pdf_path)
    except PipelineError:
        raise
    except ExtractionError as exc:
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        raise PipelineError(f"Failed to initialise ingest pipeline: {exc}") from exc

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.review_path.parent.mkdir(parents=True, exist_ok=True)
    config.reject_path.parent.mkdir(parents=True, exist_ok=True)
    config.index_path.parent.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats(total_sentences=len(sentences))
    context_history: list[str] = []
    accepted_records: list[KifRecord] = []

    with open(config.output_path, "w", encoding="utf-8") as output_handle, open(
        config.review_path, "w", encoding="utf-8"
    ) as review_handle, open(config.reject_path, "w", encoding="utf-8") as reject_handle:
        for doc_sentence in sentences:
            context = context_history[-config.context_window :] if config.context_window else []
            normalization_result, normalization_reject = _prepare_normalization(
                doc_sentence,
                context,
                normalizer,
            )

            if normalization_reject is not None:
                kif_records: list[KifRecord] = []
                review_records: list[ReviewRecord] = []
                reject_records = [normalization_reject]
            else:
                kif_records, review_records, reject_records = process_sentence(
                    doc_sentence,
                    context,
                    normalizer,
                    entity_linker,
                    sampler,
                    compiler,
                    normalization_result=normalization_result,
                    grounder=grounder,
                )

            if kif_records:
                _write_records(output_handle, kif_records)
                accepted_records.extend(kif_records)
            if review_records:
                _write_records(review_handle, review_records)
            if reject_records:
                _write_records(reject_handle, reject_records)

            _update_stats(stats, kif_records, review_records, reject_records)

            if normalization_result is not None:
                context_history.extend(normalization_result.normalized)
            if config.context_window:
                context_history = context_history[-config.context_window :]

            if config.verbose:
                for review in review_records:
                    warnings.warn(
                        f"Page {review.page_number} position {review.position} review {review.reason}: {review.detail}",
                        UserWarning,
                        stacklevel=2,
                    )
                for reject in reject_records:
                    warnings.warn(
                        f"Page {reject.page_number} position {reject.position} {reject.reason}: {reject.detail}",
                        UserWarning,
                        stacklevel=2,
                    )

    _write_index(config.index_path, accepted_records)
    return stats


def print_summary(stats: PipelineStats) -> None:
    print(
        f"Translated: {stats.translated}/{stats.total_subclaims} subclaims "
        f"({stats.coverage:.1%}) from {stats.total_sentences} sentences"
    )
    print(f"  Reviewed: {stats.reviewed}")
    print(f"  Abstained: {stats.abstained}")
    print(f"  Noise filtered: {stats.noise_filtered}")
    print(f"  Normalization ambiguous: {stats.normalization_ambiguous}")
    print(f"  Compile errors: {stats.compile_errors}")
    print(f"  Model errors: {stats.model_errors}")
    print(f"  Exceptions: {stats.exceptions}")


def main(argv: list[str] | None = None) -> int:
    config = build_pipeline(parse_args(argv))
    stats = run_pipeline(config)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
