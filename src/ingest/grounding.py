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

_CNL_TOKEN_RE = re.compile(r"\?[A-Za-z_]\w*|[A-Za-z][A-Za-z0-9_]*")
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
_EXISTENTIAL_START_RE = re.compile(r"^\s*some\s+\?[A-Za-z_]\w*\s+is-a\s+", re.IGNORECASE)
_INSTANCE_START_RE = re.compile(r"^\s*\?[A-Za-z_]\w*\s+is-a\s+", re.IGNORECASE)
_SUBCLASS_RE = re.compile(
    r"^\s*(?P<left>[A-Za-z][A-Za-z0-9_]*)\s+subclass-of\s+(?P<right>[A-Za-z][A-Za-z0-9_]*)\s*$"
)
_COPULAR_SOURCE_RE = re.compile(r"\b(?:is|are|was|were)\s+(?:an?|the|some)\b", re.IGNORECASE)
_EXISTENTIAL_SOURCE_RE = re.compile(r"\b(?:some|there\s+is|there\s+are|exists|exist)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Weak single-word argument filter
# ---------------------------------------------------------------------------
# Words that should NOT be accepted as standalone ontology class arguments in
# relation outputs.  They are common English words that frequently appear in
# doctrine text and get picked up as incidental surface-word matches but carry
# no credible ontology-entity semantics on their own.
_WEAK_ARGUMENT_WORDS: frozenset[str] = frozenset({
    # Prepositions / particles
    "on", "in", "at", "by", "to", "of", "for", "as", "or", "up",
    "out", "off", "into", "from", "with", "over", "under", "between",
    "through", "within", "without", "near", "above", "below",
    # Modal / auxiliary verbs
    "may", "can", "will", "shall", "must", "would", "could", "should", "might",
    # Adjectives / adverbs commonly surfacing as false ontology terms
    "dangerous", "likely", "unlikely", "key", "right", "left",
    "large", "larger", "largest", "small", "smaller", "smallest",
    "high", "higher", "highest", "low", "lower", "lowest",
    "long", "short", "full", "main", "major", "minor",
    "most", "least", "best", "worst", "first", "last", "next",
    "great", "greater", "greatest", "new", "old",
    "critical", "important", "significant", "effective", "current",
    "general", "primary", "secondary", "available", "possible",
    "necessary", "specific", "particular", "various", "different",
    "common", "typical", "normal", "basic", "direct", "indirect",
    "rapid", "continuous", "simultaneous", "friendly", "hostile", "decisive",
    # Determiners / quantifiers
    "all", "each", "every", "any", "both", "such", "other",
    "many", "few", "several", "more", "less", "much",
    # Generic / abstract words too vague as standalone ontology arguments
    "plan", "power", "time", "type", "kind", "part", "way", "point",
    "form", "role", "area", "case", "line", "state", "states",
    "level", "order", "place", "step", "phase", "stage", "side",
    "range", "scope", "field", "base", "core", "unit", "end",
    "goal", "task", "result", "effect", "impact", "factor",
    "method", "system", "domain", "model", "class", "value",
    "input", "output", "source", "target", "number", "size", "depth",
    "rate", "data", "fact", "risk", "cost", "loss", "gain",
    "term", "rule", "view", "mode", "means", "measure", "degree",
    "force", "need", "act", "use", "move", "turn", "shift",
    "change", "run", "set",
    "human", "enemy", "surprise", "security",
})


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
        if token.startswith("?"):
            continue
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

    @staticmethod
    def _classify_cnl(cnl: str) -> str:
        stripped = cnl.strip()
        if _EXISTENTIAL_START_RE.match(stripped):
            return "existential"
        if _INSTANCE_START_RE.match(stripped):
            return "instance"
        if " subclass-of " in stripped:
            return "subclass"
        if stripped.startswith("every ") or stripped.startswith("if "):
            return "conditional"
        if stripped.startswith("not "):
            return "negation"
        if "[" in stripped and "]" in stripped:
            return "nary"
        return "relation"

    @staticmethod
    def _supports_unary_mapping(source_text: str) -> bool:
        return bool(_COPULAR_SOURCE_RE.search(source_text))

    def _supports_existential_mapping(self, source_text: str, class_terms: Sequence[str]) -> bool:
        if not _EXISTENTIAL_SOURCE_RE.search(source_text):
            return False

        for term in class_terms:
            for phrase in _dedupe_preserve_order([_naturalize_term(term), *self._class_aliases.get(term, ())]):
                phrase_pattern = re.escape(phrase)
                existential_patterns = (
                    rf"\bthere\s+(?:is|are|exists?|exist)\b.*\b{phrase_pattern}\b",
                    rf"\bat\s+least\s+one\s+{phrase_pattern}\b",
                    rf"\bsome\s+(?:entity\s+is\s+(?:an?|the)\s+)?{phrase_pattern}\b",
                    rf"\ban?\s+instance\s+of\s+{phrase_pattern}\b",
                    rf"\b(?:an?|the)\s+{phrase_pattern}\s+exists?\b",
                )
                if any(re.search(pattern, source_text) for pattern in existential_patterns):
                    return True
        return False

    def _structural_issue(
        self,
        *,
        cnl: str,
        source_text: str,
        relation_terms: Sequence[str],
        class_terms: Sequence[str],
    ) -> tuple[str, str] | None:
        pattern = self._classify_cnl(cnl)

        if pattern == "subclass":
            match = _SUBCLASS_RE.match(cnl.strip())
            if match and match.group("left") == match.group("right"):
                return (
                    "degenerate_form",
                    f"Degenerate self-subclass statement is not accepted: {match.group('left')} subclass-of {match.group('right')}",
                )
            if len(class_terms) < 2 or not self._supports_unary_mapping(source_text):
                return (
                    "weak_grounding",
                    "Subclass output is not sufficiently supported by the source sentence structure.",
                )

        if pattern == "instance":
            if len(class_terms) < 1 or not self._supports_unary_mapping(source_text):
                return (
                    "weak_grounding",
                    "Unary instance output is not sufficiently supported by the source sentence structure.",
                )

        if pattern == "existential":
            if len(class_terms) < 1 or not self._supports_existential_mapping(source_text, class_terms):
                return (
                    "weak_grounding",
                    "Existential output is not sufficiently supported by the source sentence structure.",
                )

        if pattern == "relation" and relation_terms and len(class_terms) == 0:
            return (
                "degenerate_form",
                "Relation output has no grounded ontology arguments beyond variables.",
            )

        if pattern == "relation" and class_terms:
            rel_set = set(relation_terms)
            weak_args: list[str] = []
            for term in class_terms:
                nat = _naturalize_term(term)
                nat_words = nat.split()
                # Single-word argument in the weak set
                if len(nat_words) == 1 and nat in _WEAK_ARGUMENT_WORDS:
                    weak_args.append(term)
                # Multi-word argument with a constituent that is both weak
                # AND matches a relation term (e.g. states / UnitedStates)
                elif len(nat_words) > 1 and any(
                    w in _WEAK_ARGUMENT_WORDS and w in rel_set for w in nat_words
                ):
                    weak_args.append(term)
            if weak_args:
                return (
                    "weak_argument",
                    f"Relation argument(s) are weak single-word matches, not credible "
                    f"ontology terms: {', '.join(weak_args)}",
                )

        if pattern == "nary" and relation_terms and len(class_terms) < 3:
            return (
                "degenerate_form",
                "N-ary relation output has too few grounded ontology arguments.",
            )

        return None

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

        structural_issue = self._structural_issue(
            cnl=cnl,
            source_text=_normalise_text(subclaim_text or normalized_text or original_text),
            relation_terms=relation_terms,
            class_terms=class_terms,
        )
        ungrounded_terms = _dedupe_preserve_order(ungrounded_relations + ungrounded_classes)
        if structural_issue is not None:
            reason, detail = structural_issue
            return GroundingAssessment(
                accepted=False,
                reason=reason,
                detail=detail,
                relation_terms=relation_terms,
                class_terms=class_terms,
                terms=all_terms,
                ungrounded_terms=ungrounded_terms,
            )
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
