"""Rule-based decomposition of compound doctrine sentences into subclaims."""

from __future__ import annotations

import logging
import re

from src.fsm.cnl_fsm import CNLSampler

_LOGGER = logging.getLogger(__name__)
_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_CONJUNCTION_RE = re.compile(r"^(?:and|or|but)\s+", re.IGNORECASE)
_SEMICOLON_SPLIT_RE = re.compile(r"\s*;\s*")
_COMMA_COORD_SPLIT_RE = re.compile(r"\s*,\s*(?:and|or|but)\s+", re.IGNORECASE)
_BARE_COORD_SPLIT_RE = re.compile(r"\s+(?:and|or|but)\s+", re.IGNORECASE)
_CLAUSE_START_RE = re.compile(
    r"^(?:the|a|an|every|each|some|no|there|if|all|any|this|that|these|those|it|its|they|them|their|he|she|we|you|\?[A-Za-z_]\w*|[A-Z][a-zA-Z0-9_-]*)\b",
    re.IGNORECASE,
)
_VERBISH_TOKEN_RE = re.compile(r".*(?:s|ed|ing)$", re.IGNORECASE)
_PREDICATE_TOKENS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "shall",
    "should",
    "will",
    "would",
    "may",
    "might",
    "must",
}
_RELATIVE_CLAUSE_RE = re.compile(
    r"^(?P<head>[^,;]+?),\s*(?:which|who|that)\s+(?P<relative>[^,;]+?),\s*(?P<tail>.+)$",
    re.IGNORECASE,
)


def _normalise_sentence(text: str) -> str:
    normalised = _WHITESPACE_RE.sub(" ", text.strip())
    if not normalised:
        return ""
    normalised = _LEADING_CONJUNCTION_RE.sub("", normalised)
    normalised = normalised.strip(" ,;")
    if normalised and normalised[0].isalpha():
        normalised = normalised[0].upper() + normalised[1:]
    if normalised and normalised[-1] not in ".!?":
        normalised = f"{normalised}."
    return normalised


def _split_semicolon_list(text: str) -> list[str] | None:
    if ";" not in text:
        return None

    parts = [_normalise_sentence(part) for part in _SEMICOLON_SPLIT_RE.split(text)]
    parts = [part for part in parts if part]
    return parts if len(parts) > 1 else None


def _split_relative_clause(text: str) -> list[str] | None:
    match = _RELATIVE_CLAUSE_RE.match(text)
    if not match:
        return None

    relative = _normalise_sentence(f"{match.group('head')} {match.group('relative')}")
    tail = _normalise_sentence(f"{match.group('head')} {match.group('tail')}")
    parts = [part for part in (tail, relative) if part]
    return parts if len(parts) > 1 else None


def _looks_like_independent_clause(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 2 or not _CLAUSE_START_RE.match(text):
        return False

    predicate_tokens = [token.strip(".,!?").lower() for token in tokens[1:]]
    return any(token in _PREDICATE_TOKENS or _VERBISH_TOKEN_RE.match(token) for token in predicate_tokens)


def _split_coordinated_clauses(text: str) -> list[str] | None:
    for splitter in (_COMMA_COORD_SPLIT_RE, _BARE_COORD_SPLIT_RE):
        parts = [_normalise_sentence(part) for part in splitter.split(text)]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            continue
        if all(_looks_like_independent_clause(part.rstrip(".!?")) for part in parts):
            return parts
    return None


def _split_once(text: str) -> list[str]:
    for splitter in (_split_semicolon_list, _split_relative_clause, _split_coordinated_clauses):
        parts = splitter(text)
        if parts:
            return parts
    return [_normalise_sentence(text)]


def decompose(nl: str) -> list[str]:
    """
    Split a compound doctrine sentence into simpler, likely translatable subclaims.

    The splitter is intentionally conservative. Any subclaim that still appears
    outside the supported CNL fragment is logged and omitted.
    """
    seed = _normalise_sentence(nl)
    if not seed:
        return []

    queue = [seed]
    decomposed: list[str] = []

    while queue:
        current = queue.pop(0)
        parts = _split_once(current)
        if len(parts) == 1 and parts[0] == current:
            decomposed.append(current)
            continue
        queue = parts + queue

    translatable: list[str] = []
    for subclaim in decomposed:
        if CNLSampler.abstain_if_unsupported(subclaim):
            _LOGGER.warning("Skipping decomposed subclaim that still appears unsupported: %s", subclaim)
            continue
        translatable.append(subclaim)

    return translatable
