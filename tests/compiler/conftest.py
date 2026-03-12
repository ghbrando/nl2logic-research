import json
import pytest
from pathlib import Path

from src.compiler.compiler import CNLCompiler

SUMO_CLASSES    = Path("data/training_pairs/sumo_classes.jsonl")
SUMO_RELATIONS  = Path("data/training_pairs/sumo_relations.jsonl")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def compiler():
    return CNLCompiler()


@pytest.fixture(scope="session")
def sumo_relations():
    with open(SUMO_RELATIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="session")
def sumo_classes():
    with open(SUMO_CLASSES, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ── CNL/KIF pair helpers ───────────────────────────────────────────────────

def binary_pairs(relations: list[dict]) -> list[tuple[str, str]]:
    """
    Build (cnl, kif) pairs for binary relations from SUMO vocab.
    Uses ?arg1, ?arg2 as placeholder variables — enough to test
    the compiler's structural translation, not semantic grounding.
    """
    pairs = []
    for r in relations:
        if r.get("arity") == 2:
            term = r["term"]
            cnl = f"{term} ?arg1 ?arg2"
            kif = f"({term} ?arg1 ?arg2)"
            pairs.append((cnl, kif))
    return pairs


def nary_pairs(relations: list[dict]) -> list[tuple[str, str]]:
    """
    Build (cnl, kif) pairs for n-ary relations (arity >= 3).
    CNL uses bracket syntax: rel [ ?a1 , ?a2 , ... , ?aN ]
    """
    pairs = []
    for r in relations:
        arity = r.get("arity")
        if arity and arity >= 3:
            term = r["term"]
            vars_ = [f"?a{i}" for i in range(1, arity + 1)]
            cnl = f"{term} [ {' , '.join(vars_)} ]"
            kif = f"({term} {' '.join(vars_)})"
            pairs.append((cnl, kif))
    return pairs