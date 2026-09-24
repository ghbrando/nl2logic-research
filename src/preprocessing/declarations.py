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


# Label-free linguistic lists: subjects that refer back rather than name a
# kind, and genus heads that denote parts, members, or roles rather than kinds.
_NON_DEFINIENDUM_STARTS = frozenset({
    "this", "these", "that", "those", "there", "it", "they", "what", "while", "another",
    "such", "both", "all", "certain", "each", "many", "some", "other", "one", "following",
})
_NON_KIND_GENUS_HEADS = frozenset({
    "application", "division", "part", "subset", "one", "type", "kind", "form", "combination",
    "group", "set", "component", "aspect", "portion", "example", "member", "means", "way",
    "goal", "result", "focus", "characteristic", "contribution", "ability", "key",
})


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
               if re.match(rf"^(?:(?:a|an|the)\s+)?{re.escape(_squash(entry['alias']))}(?=\s|$)", candidate)
               and _squash(entry["evidence_cue"]) in evidence]
    if len(matches) > 1:
        raise ValueError("Multiple provisional declarations match one candidate")
    return matches[0] if matches else None


def propose_declaration(candidate_text: str, known_classes: set[str] | frozenset[str],
                        evidence_sentence: str) -> dict | None:
    """Propose a new class name from a copular definition, using the source alone.

    The symbol joins the defined phrase and the head of its article-introduced
    genus noun phrase ("Biometrics is the process of ..." -> BiometricsProcess;
    "Risk management is the Army's primary process for ..." ->
    RiskManagementProcess), so it names the sense the source states. It asserts
    no parent. Pronoun or demonstrative subjects, adjective predicates,
    coordinated or non-kind genus heads, and existing classes are declined.
    """
    from src.ingest.grounding import _normalise_text, genus_noun_phrase

    match = _DEFINITION_FRAME.match(" ".join(candidate_text.split()))
    if match is None:
        return None
    subject = re.sub(r"^(?:a|an|the)\s+", "", match.group("subject"), flags=re.IGNORECASE)
    if not subject or subject.split()[0].casefold() in _NON_DEFINIENDUM_STARTS:
        return None
    sentence = _normalise_text(evidence_sentence)
    frame = re.match(rf"(?:(?:a|an|the)\s+)?{re.escape(_normalise_text(subject))}\s+(?:is|are)\s+", sentence)
    if frame is None:
        return None
    has_article, genus, _ = genus_noun_phrase(sentence[frame.end():])
    if (not has_article or not genus or any(token in {"and", "or", "nor", "but"} for token in genus)
            or genus[-1] in _NON_KIND_GENUS_HEADS or genus[-1].removesuffix("s") in _NON_KIND_GENUS_HEADS
            or not genus[-1].isalpha() or len(genus[-1]) < 3):
        return None
    stem, suffix = _pascal(subject), _pascal(genus[-1])
    symbol = stem if stem.endswith(suffix) else stem + suffix
    if not stem or _SYMBOL.fullmatch(symbol) is None or symbol in known_classes or stem in known_classes:
        return None
    article = f" {match.group('article')}" if match.group("article") else ""
    return {
        "alias": subject,
        "evidence_cue": f"{match.group('copula')}{article}",
        "symbol": symbol,
        "declaration": f"(instance {symbol} Class)",
        "genus_phrase": " ".join(genus),
        "sense": f"what the source defines as '{subject}' with genus '{' '.join(genus)}'",
    }


def parent_candidates(declaration: dict, parent_pool: list[str], known_classes) -> list[str]:
    """Offer the fixed parent pool plus existing classes named by the genus phrase.

    Candidates are options for the decoder, not assertions; the passage-only
    gate still has to find the chosen parent stated as the genus.
    """
    # Registries proposed before genus phrases were recorded kept the head as the cue's last word.
    genus = (declaration.get("genus_phrase") or declaration["evidence_cue"].split()[-1]).split()
    named = [_pascal(" ".join(genus)), _pascal(genus[-1])]
    return list(dict.fromkeys([*parent_pool, *(term for term in named if term in known_classes)]))
