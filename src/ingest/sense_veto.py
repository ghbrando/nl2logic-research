"""Model sense veto for claims the passage-only gate has already accepted.

The gate checks that the passage names the parent as the genus. It cannot tell
whether the passage uses that word in the parent class's sense ("the product
of analysis" is not a SUMO Product, which is manufactured). The veto shows a
model the evidence sentence, the claim, and the parent's ontology gloss, and
compares the likelihood of the claim against its CNL negation. It can only
remove accepted claims; it never adds one.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SUMO_CLASSES = ROOT / "data/training_pairs/sumo_classes.jsonl"
_DOCTRINE_KIF = ROOT / "data/ontology/doctrine_domain.kif"
_DOC_RE = re.compile(r'\(documentation\s+([A-Za-z0-9_]+)\s+EnglishLanguage\s+"(.*?)"\)', re.DOTALL)


def _first_sentence(text: str) -> str:
    text = " ".join(text.replace("&%", "").replace(";;", "").split())
    return re.split(r"(?<=[.;])\s", text, maxsplit=1)[0]


@lru_cache(maxsize=1)
def _glosses() -> dict[str, str]:
    glosses = {}
    for line in _SUMO_CLASSES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("doc"):
                glosses[row["term"]] = _first_sentence(row["doc"])
    for term, doc in _DOC_RE.findall(_DOCTRINE_KIF.read_text(encoding="utf-8")):
        glosses[term] = _first_sentence(doc)
    return glosses


def parent_gloss(parent: str) -> str:
    return _glosses().get(parent, "No ontology definition available.")


def veto_prompt(sentence: str, symbol: str, parent: str) -> str:
    """Encoder text without the "translate to CNL: " prefix that training adds."""
    return (
        "Check whether the source uses the parent in the ontology's sense.\n"
        f"Source: {sentence}\n"
        f"Claim: {symbol} subclass-of {parent}\n"
        f"Parent meaning: {parent_gloss(parent)}"
    )


def accept_target(symbol: str, parent: str) -> str:
    return f"{symbol} subclass-of {parent}"


def reject_target(symbol: str, parent: str) -> str:
    return f"not {symbol} subclass-of {parent}"
