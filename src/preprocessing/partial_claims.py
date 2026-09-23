"""Conservative, source-preserving candidate extraction for doctrine definitions.

This stage only selects evidence to send downstream. It never asserts an
ontology relation or chooses a sense for an ambiguous source term.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

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
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


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


def extract_paragraph_candidate(source_excerpt: str, first_sentence: str) -> tuple[PartialClaimCandidate, list[dict]]:
    """Scan exact paragraph sentences and return the first conservative head.

    The existing PDF first-sentence extraction anchors the first boundary.
    Later boundaries are split on punctuation followed by a capitalized word;
    false splits are possible and remain a prototype limitation.
    """
    if not first_sentence or not source_excerpt.startswith(first_sentence):
        raise ValueError("First sentence must be the exact paragraph prefix")
    spans = [(0, len(first_sentence))]
    tail_start = len(first_sentence)
    while tail_start < len(source_excerpt) and source_excerpt[tail_start].isspace():
        tail_start += 1
    if tail_start < len(source_excerpt):
        tail = source_excerpt[tail_start:]
        boundaries = [0]
        for match in _SENTENCE_BOUNDARY.finditer(tail):
            preceding = tail[:match.start()]
            if preceding.endswith(("U.S.", "U.K.", "e.g.", "i.e.")):
                continue
            boundaries.append(match.end())
        boundaries.append(len(tail))
        for start, end in zip(boundaries, boundaries[1:]):
            raw_start = tail_start + start
            raw_end = tail_start + end
            while raw_end > raw_start and source_excerpt[raw_end - 1].isspace():
                raw_end -= 1
            if raw_start < raw_end:
                spans.append((raw_start, raw_end))
    screened: list[dict] = []
    for start, end in spans:
        sentence = source_excerpt[start:end]
        result = extract_definition_head(sentence, sentence)
        screened.append({"start": start, "end": end, "status": result.status, "reason": result.reason})
        if result.status == "candidate":
            return replace(result, start=start + result.start, end=start + result.end), screened
    return PartialClaimCandidate("abstain", "no_supported_definition_head", None, None, None, None, ()), screened
