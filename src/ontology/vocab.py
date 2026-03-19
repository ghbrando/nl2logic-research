"""Shared closed-vocabulary source for the NL2Logic doctrine pipeline.

Provides the authoritative answer to: what class/relation terms are allowed
in the live doctrine pipeline?

The closed vocabulary is the union of:
  - Base SUMO classes/relations from data/training_pairs/ JSONL files
  - Doctrine extension classes defined in data/ontology/doctrine_domain.kif
    via (subclass X Y) declarations

Use this module instead of loading sumo_classes.jsonl directly anywhere
in the live pipeline.  Keeping one loader here means doctrine extensions
automatically propagate to the compiler, FSM, grounder, and entity linker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUMO_CLASSES = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_DEFAULT_SUMO_RELATIONS = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"
_DEFAULT_DOCTRINE_KIF = _REPO_ROOT / "data" / "ontology" / "doctrine_domain.kif"

# Matches the left-hand (child) term in a KIF subclass assertion.
# Ignores documentation lines, comments (;;), and blank lines.
_KIF_SUBCLASS_RE = re.compile(r"^\s*\(subclass\s+(\S+)\s+\S+\s*\)")

# Captures both child (group 1) and parent (group 2) from a subclass assertion.
_KIF_SUBCLASS_PAIR_RE = re.compile(r"^\s*\(subclass\s+(\S+)\s+(\S+)\s*\)")


def extract_kif_class_terms(kif_path: Path) -> set[str]:
    """Extract child class names from (subclass X Y) declarations in a KIF file.

    Only the left-hand child class of each subclass assertion is returned.
    Parent classes are assumed to already exist in the base SUMO vocabulary.
    """
    terms: set[str] = set()
    for line in kif_path.read_text(encoding="utf-8").splitlines():
        m = _KIF_SUBCLASS_RE.match(line)
        if m:
            terms.add(m.group(1))
    return terms


def extract_kif_subclass_pairs(kif_path: Path) -> dict[str, str]:
    """Return a {child: parent} mapping from (subclass X Y) declarations in a KIF file."""
    pairs: dict[str, str] = {}
    for line in kif_path.read_text(encoding="utf-8").splitlines():
        m = _KIF_SUBCLASS_PAIR_RE.match(line)
        if m:
            pairs[m.group(1)] = m.group(2)
    return pairs


def load_doctrine_subclass_pairs(
    doctrine_kif: Path = _DEFAULT_DOCTRINE_KIF,
) -> dict[str, str]:
    """Return the child→parent edge map from doctrine_domain.kif."""
    return extract_kif_subclass_pairs(doctrine_kif)


def load_closed_class_terms(
    sumo_classes: Path = _DEFAULT_SUMO_CLASSES,
    doctrine_kif: Path | None = _DEFAULT_DOCTRINE_KIF,
) -> set[str]:
    """Return the closed set of allowed class terms for the doctrine pipeline.

    Includes:
    - All class names from the SUMO JSONL vocab file
    - All child class names defined via (subclass X Y) in the doctrine KIF file

    Parameters
    ----------
    sumo_classes:
        Path to the SUMO class JSONL file.  Defaults to the repo's standard
        data/training_pairs/sumo_classes.jsonl.
    doctrine_kif:
        Path to a doctrine KIF extension file, or None to skip.  Defaults to
        data/ontology/doctrine_domain.kif.
    """
    terms: set[str] = set()
    with open(sumo_classes, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terms.add(json.loads(line)["term"])
    if doctrine_kif is not None and doctrine_kif.exists():
        terms |= extract_kif_class_terms(doctrine_kif)
    return terms


def load_closed_relation_terms(
    sumo_relations: Path = _DEFAULT_SUMO_RELATIONS,
    doctrine_kif: Path | None = _DEFAULT_DOCTRINE_KIF,
) -> dict[str, int | None]:
    """Return the closed set of allowed relation terms with their arity.

    Currently doctrine_domain.kif defines no new relations; the doctrine_kif
    parameter is accepted for API consistency and future use.

    Parameters
    ----------
    sumo_relations:
        Path to the SUMO relations JSONL file.
    doctrine_kif:
        Accepted but currently unused (no relation definitions in doctrine KIF).
    """
    relations: dict[str, int | None] = {}
    with open(sumo_relations, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                relations[r["term"]] = r.get("arity")
    return relations
