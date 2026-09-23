"""Conservative, source-preserving candidate extraction for doctrine definitions.

This stage only selects evidence to send downstream. It never asserts an
ontology relation or chooses a sense for an ambiguous source term.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.fsm.cnl_fsm import CNLSampler

_DEFINITION = re.compile(
    r"^\s*(?P<subject>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,5}?)"
    r"\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?(?P<head>[A-Za-z][A-Za-z'-]*)\b",
    re.IGNORECASE,
)
_SCOPE_START = re.compile(r"^(?:within|under|during|when|if|although|according|for)\b", re.IGNORECASE)
_UNSAFE_SENTENCE = re.compile(r"\b(?:or|either|neither|nor|unless|except|if|not|no|may|might|can|must)\b", re.IGNORECASE)
_NON_KIND_HEADS = frozenset({
    "application", "division", "part", "subset", "one", "type", "kind", "form",
    "processing", "combination", "group", "set", "component", "aspect",
})


@dataclass(frozen=True)
class PartialClaimCandidate:
    status: str
    reason: str | None
    source_span: str | None
    start: int | None
    end: int | None
    candidate_text: str | None
    gate_reasons: tuple[str, ...]


def extract_definition_head(source_excerpt: str, source_sentence: str) -> PartialClaimCandidate:
    """Select a short exact copular head from the first sentence, or abstain.

    The full sentence remains available for later semantic review. A selected
    head is not a formalization: it may be too vague or use an ambiguous sense.
    """
    if not source_sentence or not source_excerpt.startswith(source_sentence):
        raise ValueError("Source sentence must be the exact paragraph prefix")

    def abstain(reason: str) -> PartialClaimCandidate:
        return PartialClaimCandidate("abstain", reason, None, None, None, None, ())

    if _SCOPE_START.match(source_sentence):
        return abstain("leading_scope")
    if _UNSAFE_SENTENCE.search(source_sentence):
        return abstain("condition_negation_or_disjunction")
    match = _DEFINITION.match(source_sentence)
    if match is None:
        return abstain("no_simple_definition_head")
    head = match.group("head").casefold()
    if head in _NON_KIND_HEADS or head.endswith("ed"):
        return abstain("non_kind_predicate")
    start = match.start("subject")
    end = match.end("head")
    raw_span = source_sentence[start:end]
    candidate_text = " ".join(raw_span.split())
    reasons = tuple(CNLSampler._unsupported_reasons(candidate_text))
    return PartialClaimCandidate(
        "candidate", None, raw_span, start, end, candidate_text, reasons
    )
