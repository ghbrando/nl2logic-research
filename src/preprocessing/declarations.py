"""Isolated provisional class declarations for development-only inference."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.ontology.vocab import load_closed_class_terms

_SYMBOL = re.compile(r"[A-Z][A-Za-z0-9]*\Z")


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


ORACLE_STATUS = "ai_reviewed_development_oracle_not_evaluation"
PROPOSED_STATUS = "source_only_proposed_not_reviewed"
_DEFINITION_FRAME = re.compile(
    r"^(?P<subject>.+?)\s+(?P<copula>is|are)\s+(?:(?P<article>a|an|the)\s+)?(?P<head>[A-Za-z][A-Za-z'-]*)$",
    re.IGNORECASE,
)


def _pascal(text: str) -> str:
    return "".join(word[:1].upper() + word[1:].lower() for word in re.findall(r"[A-Za-z0-9]+", text))


def load_declaration_registry(path: Path, *, source_packet_sha256: str | None = None,
                              sources_sha256: str | None = None) -> dict:
    """Load an isolated registry tied to the exact inputs it was made for.

    An oracle registry is tied to the reviewed development packet. A proposed
    registry is generated from unlabeled sources alone and is tied to them.
    """
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry["status"] == ORACLE_STATUS:
        tied = source_packet_sha256 is not None and registry.get("source_packet_sha256") == source_packet_sha256
    elif registry["status"] == PROPOSED_STATUS:
        tied = sources_sha256 is not None and registry.get("sources_sha256") == sources_sha256
    else:
        tied = False
    if not tied:
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


def propose_declaration(candidate_text: str, known_classes: set[str] | frozenset[str]) -> dict | None:
    """Propose a new class name from a copular definition head, using the source alone.

    The symbol joins the defined phrase and its head noun ("Biometrics is the
    process" -> BiometricsProcess), so it names the sense the source states.
    It asserts no parent. Existing ontology classes are not redeclared.
    """
    match = _DEFINITION_FRAME.match(" ".join(candidate_text.split()))
    if match is None:
        return None
    subject = re.sub(r"^(?:a|an|the)\s+", "", match.group("subject"), flags=re.IGNORECASE)
    head = match.group("head")
    stem, suffix = _pascal(subject), _pascal(head)
    symbol = stem if stem.endswith(suffix) else stem + suffix
    if not stem or _SYMBOL.fullmatch(symbol) is None or symbol in known_classes or stem in known_classes:
        return None
    article = f" {match.group('article')}" if match.group("article") else ""
    return {
        "alias": subject,
        "evidence_cue": f"{match.group('copula')}{article} {head}",
        "symbol": symbol,
        "declaration": f"(instance {symbol} Class)",
        "sense": f"what the source defines as '{subject}' with head '{head}'",
    }


def parent_candidates(declaration: dict, parent_pool: list[str], known_classes) -> list[str]:
    """Offer the fixed parent pool plus an existing class named by the head noun.

    Candidates are options for the decoder, not assertions; the passage-only
    gate still has to find the chosen parent stated as the genus.
    """
    head = declaration["evidence_cue"].split()[-1]
    named = _pascal(head)
    return list(dict.fromkeys([*parent_pool, *([named] if named in known_classes else [])]))
