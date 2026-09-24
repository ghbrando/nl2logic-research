"""Deterministic retrieval of prior ontology statements as decoder context.

These statements are background axioms, not source evidence for a new claim.
The module does not add terms to the closed decoder vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.ontology.vocab import load_closed_class_terms

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY = ROOT / "data/ontology/doctrine_domain.kif"
_SUBCLASS = re.compile(r"^\s*\(subclass\s+([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\s*\)\s*$")
_PASCAL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN = re.compile(r"[a-z0-9]+")
_GENERIC = frozenset({"a", "an", "the", "is", "are", "of", "and", "to", "intelligence", "product", "process", "military", "information"})


def _words(text: str) -> set[str]:
    return set(_TOKEN.findall(_PASCAL_BOUNDARY.sub(" ", text).lower()))


@dataclass(frozen=True)
class MemoryStatement:
    statement_id: str
    kif: str
    child: str
    parent: str
    source_path: str
    source_line: int
    source_sha256: str
    kind: str = "ontology_background"


def load_ontology_memory(path: Path = DEFAULT_ONTOLOGY) -> list[MemoryStatement]:
    source_sha = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    known_classes = load_closed_class_terms()
    statements = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _SUBCLASS.fullmatch(line)
        if match is None:
            continue
        child, parent = match.groups()
        if child not in known_classes or parent not in known_classes:
            raise ValueError(f"Unrecognized ontology term on line {line_number}")
        statements.append(MemoryStatement(
            statement_id=f"doctrine-domain:{line_number}",
            kif=f"(subclass {child} {parent})",
            child=child,
            parent=parent,
            source_path="data/ontology/doctrine_domain.kif",
            source_line=line_number,
            source_sha256=source_sha,
        ))
    return statements


def retrieve_memory(source_text: str, statements: list[MemoryStatement], *, limit: int = 2) -> list[MemoryStatement]:
    """Rank literal child-name overlap; omit generic-only matches."""
    if limit < 0:
        raise ValueError("Memory limit cannot be negative")
    source_words = _words(source_text)
    ranked = []
    for statement in statements:
        child_words = _words(statement.child)
        overlap = source_words & child_words
        specific = overlap - _GENERIC
        if not specific:
            continue
        ranked.append(((-len(overlap), len(child_words - source_words), statement.source_line), statement))
    ranked.sort(key=lambda item: item[0])
    return [statement for _, statement in ranked[:limit]]


def render_memory_prompt(source_prompt: str, statements: list[MemoryStatement]) -> str:
    if not statements:
        return source_prompt
    labels = {"ontology_background": "Prior ontology statement", "prior_accepted_claim": "Prior accepted claim"}
    context = "\n".join(f"{labels[statement.kind]}: {statement.kif}" for statement in statements)
    return (
        f"{source_prompt}\n{context}\n"
        "Prior statements are context only. Formalize only what the current source states."
    )


def load_prior_claim_memory(scored_path: Path, *, mode: str = "rules") -> list[MemoryStatement]:
    """Load claims a previous, separate run accepted through the passage-only gate.

    They are context for the encoder only. They are never grounding evidence and
    carry the provenance of the run that accepted them.
    """
    text = scored_path.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    statements = []
    for line_number, line in enumerate(text.splitlines(), 1):
        row = json.loads(line) if line.strip() else None
        if not row or row.get("mode") != mode or row.get("declared_gate_accepted") is not True:
            continue
        match = re.fullmatch(r"\(subclass ([A-Za-z][A-Za-z0-9_]*) ([A-Za-z][A-Za-z0-9_]*)\)", row["compiled_kif"])
        if match is None:
            raise ValueError(f"Prior accepted claim on line {line_number} is not a subclass claim")
        child, parent = match.groups()
        statements.append(MemoryStatement(
            statement_id=f"prior-claim:{row['record_id']}",
            kif=row["compiled_kif"], child=child, parent=parent,
            source_path=scored_path.as_posix(), source_line=line_number,
            source_sha256=source_sha, kind="prior_accepted_claim",
        ))
    return statements
