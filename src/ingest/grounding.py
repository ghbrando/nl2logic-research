"""Grounding helpers for precision-first doctrine ingestion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.fsm.cnl_fsm import _PASCAL_BOUNDARY_RE, _PROMPT_CLASS_ALIASES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSES_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_RELATIONS_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"

_CNL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NORMALIZED_TEXT_RE = re.compile(r"[^a-z0-9\s]+")
_RESERVED_CNL_TOKENS = {
    "a",
    "and",
    "every",
    "if",
    "implies",
    "is",
    "no",
    "not",
    "of",
    "some",
    "subclass",
    "then",
}


def _load_terms(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line)["term"] for line in handle if line.strip()]


def _normalise_text(text: str) -> str:
    text = _PASCAL_BOUNDARY_RE.sub(" ", text.replace("_", " "))
    text = _NORMALIZED_TEXT_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _naturalize_term(term: str) -> str:
    return _normalise_text(term)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def extract_cnl_terms(cnl: str) -> tuple[list[str], list[str], list[str]]:
    """Return relation terms, class terms, and all ontology terms from canonical CNL."""
    relation_terms: list[str] = []
    class_terms: list[str] = []
    all_terms: list[str] = []

    for token in _CNL_TOKEN_RE.findall(cnl):
        lower = token.lower()
        if lower in _RESERVED_CNL_TOKENS:
            continue

        all_terms.append(token)
        if token[0].isupper():
            class_terms.append(token)
        elif token[0].islower():
            relation_terms.append(token)

    return (
        _dedupe_preserve_order(relation_terms),
        _dedupe_preserve_order(class_terms),
        _dedupe_preserve_order(all_terms),
    )


@dataclass(frozen=True)
class GroundingAssessment:
    accepted: bool
    reason: str | None
    detail: str
    relation_terms: list[str]
    class_terms: list[str]
    terms: list[str]
    ungrounded_terms: list[str]


class DoctrineGrounder:
    """Lexical grounding validator for accepted doctrine facts."""

    def __init__(
        self,
        *,
        classes_path: Path = _CLASSES_PATH,
        relations_path: Path = _RELATIONS_PATH,
    ) -> None:
        self._classes = set(_load_terms(classes_path))
        self._relations = set(_load_terms(relations_path))
        self._class_aliases = self._build_class_aliases()

    @staticmethod
    def _build_class_aliases() -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        for phrase, terms in _PROMPT_CLASS_ALIASES.items():
            normalized_phrase = _normalise_text(phrase)
            for term in terms:
                aliases.setdefault(term, []).append(normalized_phrase)
        return aliases

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        if not text or not phrase:
            return False
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    def _grounded_class(self, term: str, contexts: Sequence[str]) -> bool:
        normalized_term = _naturalize_term(term)
        candidates = [normalized_term, *self._class_aliases.get(term, ())]
        return any(self._contains_phrase(context, candidate) for context in contexts for candidate in candidates)

    def _grounded_relation(self, term: str, contexts: Sequence[str]) -> bool:
        normalized_term = _naturalize_term(term)
        return any(self._contains_phrase(context, normalized_term) for context in contexts)

    def assess(
        self,
        *,
        cnl: str,
        original_text: str,
        normalized_text: str,
        linked_text: str,
        subclaim_text: str,
    ) -> GroundingAssessment:
        relation_terms, class_terms, all_terms = extract_cnl_terms(cnl)
        contexts = _dedupe_preserve_order(
            _normalise_text(text)
            for text in (original_text, normalized_text, linked_text, subclaim_text)
            if text and text.strip()
        )

        ungrounded_relations = [
            term
            for term in relation_terms
            if term in self._relations and not self._grounded_relation(term, contexts)
        ]
        ungrounded_classes = [
            term
            for term in class_terms
            if term in self._classes and not self._grounded_class(term, contexts)
        ]

        ungrounded_terms = _dedupe_preserve_order(ungrounded_relations + ungrounded_classes)
        if ungrounded_relations:
            return GroundingAssessment(
                accepted=False,
                reason="ungrounded_relation",
                detail=f"Generated relation terms are not lexically supported by the source text: {', '.join(ungrounded_relations)}",
                relation_terms=relation_terms,
                class_terms=class_terms,
                terms=all_terms,
                ungrounded_terms=ungrounded_terms,
            )
        if ungrounded_classes:
            return GroundingAssessment(
                accepted=False,
                reason="ungrounded_class_terms",
                detail=f"Generated class terms are not lexically supported by the source text: {', '.join(ungrounded_classes)}",
                relation_terms=relation_terms,
                class_terms=class_terms,
                terms=all_terms,
                ungrounded_terms=ungrounded_terms,
            )
        return GroundingAssessment(
            accepted=True,
            reason=None,
            detail="All ontology terms are lexically grounded in the source text.",
            relation_terms=relation_terms,
            class_terms=class_terms,
            terms=all_terms,
            ungrounded_terms=[],
        )
