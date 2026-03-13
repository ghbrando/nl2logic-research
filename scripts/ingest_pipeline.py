from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.compiler.compiler import CNLCompiler
from src.eval.model_predictor import DEFAULT_MODEL_PATH, load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.ingest.extractor import DocSentence, ExtractionError, extract_sentences
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
DEFAULT_REJECT_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_rejects.jsonl"
DEFAULT_CONTEXT_WINDOW = 3


class PipelineError(Exception):
    """Raised when the ingest pipeline cannot be initialised safely."""


@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path
    output_path: Path
    reject_path: Path
    model_path: Path
    normalizer: str
    anthropic_api_key: str | None
    context_window: int
    verbose: bool


@dataclass(frozen=True)
class KifRecord:
    kif: str
    cnl: str
    nl: str
    original: str
    page_number: int
    position: int
    section: str
    pattern: str


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
    abstained: int = 0
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
        "--rejects",
        type=Path,
        default=DEFAULT_REJECT_PATH,
        help=f"Reject-log JSONL output path (default: {DEFAULT_REJECT_PATH})",
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
        reject_path=Path(args.rejects),
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


def process_sentence(
    doc_sentence: DocSentence,
    context: list[str],
    normalizer: Callable[[str, list[str]], str] | None,
    entity_linker: EntityLinker,
    sampler: CNLSampler,
    compiler: CNLCompiler,
    *,
    normalization_result: NormalizationResult | None = None,
) -> tuple[list[KifRecord], list[RejectRecord]]:
    try:
        result = normalization_result
        if result is None:
            result, normalization_reject = _prepare_normalization(doc_sentence, context, normalizer)
            if normalization_reject is not None or result is None:
                return [], [normalization_reject or _reject(
                    doc_sentence,
                    subclaim=doc_sentence.sentence,
                    reason="exception",
                    detail="Normalization failed unexpectedly.",
                )]
        kif_records: list[KifRecord] = []
        reject_records: list[RejectRecord] = []

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
            return kif_records, reject_records

        for normalized_sentence in result.normalized:
            linked_sentence = entity_linker.link(normalized_sentence)
            subclaims = decompose(linked_sentence)
            if not subclaims:
                reject_records.append(
                    _reject(
                        doc_sentence,
                        subclaim=linked_sentence,
                        reason="abstained",
                        detail="No translatable subclaims after decomposition.",
                    )
                )
                continue

            for subclaim in subclaims:
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

                kif_records.append(
                    KifRecord(
                        kif=kif,
                        cnl=cnl,
                        nl=subclaim,
                        original=doc_sentence.sentence,
                        page_number=doc_sentence.page_number,
                        position=doc_sentence.position,
                        section=doc_sentence.section,
                        pattern=_get_sampler_pattern(sampler),
                    )
                )

        return kif_records, reject_records
    except Exception as exc:
        return [], [
            _reject(
                doc_sentence,
                subclaim=doc_sentence.sentence,
                reason="exception",
                detail=str(exc),
            )
        ]


def _write_records(handle, records: list[Any]) -> None:
    for record in records:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    handle.flush()


def _update_stats(stats: PipelineStats, kif_records: list[KifRecord], reject_records: list[RejectRecord]) -> None:
    stats.translated += len(kif_records)
    stats.total_subclaims += len(kif_records)

    for reject in reject_records:
        if reject.reason == "abstained":
            stats.abstained += 1
            stats.total_subclaims += 1
        elif reject.reason == "normalization_ambiguous":
            stats.normalization_ambiguous += 1
        elif reject.reason == "compile_error":
            stats.compile_errors += 1
            stats.total_subclaims += 1
        elif reject.reason == "model_error":
            stats.model_errors += 1
            stats.total_subclaims += 1
        elif reject.reason == "exception":
            stats.exceptions += 1
            stats.total_subclaims += 1


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
        sentences = extract_sentences(config.pdf_path)
    except PipelineError:
        raise
    except ExtractionError as exc:
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        raise PipelineError(f"Failed to initialise ingest pipeline: {exc}") from exc

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.reject_path.parent.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats(total_sentences=len(sentences))
    context_history: list[str] = []

    with open(config.output_path, "w", encoding="utf-8") as output_handle, open(
        config.reject_path, "w", encoding="utf-8"
    ) as reject_handle:
        for doc_sentence in sentences:
            context = context_history[-config.context_window :] if config.context_window else []
            normalization_result, normalization_reject = _prepare_normalization(
                doc_sentence,
                context,
                normalizer,
            )

            if normalization_reject is not None:
                kif_records = []
                reject_records = [normalization_reject]
            else:
                kif_records, reject_records = process_sentence(
                    doc_sentence,
                    context,
                    normalizer,
                    entity_linker,
                    sampler,
                    compiler,
                    normalization_result=normalization_result,
                )

            if kif_records:
                _write_records(output_handle, kif_records)
            if reject_records:
                _write_records(reject_handle, reject_records)

            _update_stats(stats, kif_records, reject_records)

            if normalization_result is not None:
                context_history.extend(normalization_result.normalized)
            if config.context_window:
                context_history = context_history[-config.context_window :]

            if config.verbose:
                for reject in reject_records:
                    warnings.warn(
                        f"Page {reject.page_number} position {reject.position} {reject.reason}: {reject.detail}",
                        UserWarning,
                        stacklevel=2,
                    )

    return stats


def print_summary(stats: PipelineStats) -> None:
    print(
        f"Translated: {stats.translated}/{stats.total_subclaims} subclaims "
        f"({stats.coverage:.1%}) from {stats.total_sentences} sentences"
    )
    print(f"  Abstained: {stats.abstained}")
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
