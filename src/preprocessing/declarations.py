"""Isolated provisional class declarations for development-only inference."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.ontology.vocab import load_closed_class_terms

_SYMBOL = re.compile(r"[A-Z][A-Za-z0-9]*\Z")


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


def load_declaration_registry(path: Path, *, source_packet_sha256: str) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if (registry["status"] != "ai_reviewed_development_oracle_not_evaluation"
            or registry["source_packet_sha256"] != source_packet_sha256):
        raise ValueError("Provisional declaration registry is not tied to this development source")
    known = load_closed_class_terms()
    parents = registry["existing_parent_pool"]
    if not parents or len(parents) != len(set(parents)) or any(term not in known for term in parents):
        raise ValueError("Parent pool must contain distinct existing ontology classes")
    seen_aliases, seen_symbols = set(), set()
    for entry in registry["declarations"]:
        alias, cue, symbol = entry["alias"], entry["evidence_cue"], entry["symbol"]
        if (not alias or not cue or _SYMBOL.fullmatch(symbol) is None or symbol in known
                or symbol in seen_symbols or _squash(alias) in seen_aliases
                or entry["declaration"] != f"(instance {symbol} Class)"
                or any(key in entry for key in ("cnl", "kif", "decision", "parent", "subclass"))):
            raise ValueError("Invalid, duplicate, or assertion-bearing provisional declaration")
        seen_aliases.add(_squash(alias))
        seen_symbols.add(symbol)
    return registry


def match_declaration(candidate_text: str, evidence_sentence: str, registry: dict) -> dict | None:
    candidate = _squash(candidate_text)
    evidence = _squash(evidence_sentence)
    matches = [entry for entry in registry["declarations"]
               if re.match(rf"^{re.escape(_squash(entry['alias']))}(?=\s|$)", candidate)
               and _squash(entry["evidence_cue"]) in evidence]
    if len(matches) > 1:
        raise ValueError("Multiple provisional declarations match one candidate")
    return matches[0] if matches else None
