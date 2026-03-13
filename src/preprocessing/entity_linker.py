"""String-matching entity linker for doctrine acronyms and SUMO terms."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSES_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_RELATIONS_PATH = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"

_DEFAULT_ABBREVIATIONS = {
    "OPORD": "Order",
    "CO": "Organization",
    "BN": "MilitaryUnit",
    "BDE": "MilitaryOrganization",
    "CCIR": "Communication",
    "ISR": "Reconnaissance",
    "C2": "Communication",
    "AO": "Area",
    "OBJ": "MilitaryProcess",
}


def _load_terms(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line)["term"] for line in handle if line.strip()]


class EntityLinker:
    """
    Link doctrine acronyms and simple surface-form mentions to SUMO terms.

    The linker is deliberately string-based only. It uses the extracted SUMO
    class/relation vocab plus a small hardcoded acronym table and performs
    case-insensitive token replacement without invoking a model.
    """

    def __init__(
        self,
        *,
        classes_path: Path = _CLASSES_PATH,
        relations_path: Path = _RELATIONS_PATH,
        abbreviation_map: dict[str, str] | None = None,
    ) -> None:
        self._classes = set(_load_terms(classes_path))
        self._relations = set(_load_terms(relations_path))
        self._vocab = self._classes | self._relations
        self._lookup = self._build_lookup(abbreviation_map or {})
        self._pattern = self._build_pattern(self._lookup)

    def _build_lookup(self, extra_abbreviations: dict[str, str]) -> dict[str, str]:
        lookup: dict[str, str] = {}

        for term in sorted(self._vocab, key=len, reverse=True):
            lookup.setdefault(term.casefold(), term)

        abbreviations = dict(_DEFAULT_ABBREVIATIONS)
        abbreviations.update(extra_abbreviations)
        for source, target in abbreviations.items():
            if target not in self._vocab:
                _LOGGER.debug(
                    "Skipping abbreviation %s -> %s because %s is not present in the local SUMO vocab.",
                    source,
                    target,
                    target,
                )
                continue
            lookup[source.casefold()] = target

        return lookup

    @staticmethod
    def _build_pattern(lookup: dict[str, str]) -> re.Pattern[str] | None:
        if not lookup:
            return None

        alternatives = sorted(lookup, key=len, reverse=True)
        pattern = "|".join(re.escape(key) for key in alternatives)
        return re.compile(rf"(?<![A-Za-z0-9_-])(?:{pattern})(?![A-Za-z0-9_-])", re.IGNORECASE)

    def link(self, nl: str) -> str:
        """Replace recognized tokens with canonical SUMO class/relation terms."""
        if not nl or self._pattern is None:
            return nl

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            replacement = self._lookup.get(token.casefold())
            if replacement is None or token == replacement:
                return token
            return replacement

        return self._pattern.sub(replace, nl)
