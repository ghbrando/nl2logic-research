import json
import pytest
from pathlib import Path

SUMO_RELATIONS = Path("data/training_pairs/sumo_relations.jsonl")


def _load_relations() -> list[dict]:
    with open(SUMO_RELATIONS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _binary_pairs() -> list[tuple[str, str]]:
    pairs = []
    for r in _load_relations():
        if r.get("arity") == 2:
            term = r["term"]
            pairs.append((f"{term} ?arg1 ?arg2", f"({term} ?arg1 ?arg2)"))
        if len(pairs) >= 30:
            break
    return pairs


# ── Tests ──────────────────────────────────────────────────────────────────

class TestBinaryRelations:

    # Military-critical spot checks — these must always pass
    def test_agent(self, compiler):
        assert compiler.compile("agent ?attack ?soldier") == "(agent ?attack ?soldier)"

    def test_patient(self, compiler):
        assert compiler.compile("patient ?mission ?target") == "(patient ?mission ?target)"

    def test_instrument(self, compiler):
        assert compiler.compile("instrument ?strike ?weapon") == "(instrument ?strike ?weapon)"

    def test_destination(self, compiler):
        assert compiler.compile("destination ?movement ?location") == "(destination ?movement ?location)"

    def test_origin(self, compiler):
        assert compiler.compile("origin ?patrol ?basecamp") == "(origin ?patrol ?basecamp)"

    def test_member(self, compiler):
        assert compiler.compile("member ?soldier ?unit") == "(member ?soldier ?unit)"

    # Argument variables are preserved exactly as given
    def test_variable_names_preserved(self, compiler):
        assert compiler.compile("agent ?x ?y") == "(agent ?x ?y)"
        assert compiler.compile("agent ?attack ?commandingOfficer") == "(agent ?attack ?commandingOfficer)"

    # SUMO-derived parametrized coverage
    @pytest.mark.parametrize("cnl,expected", _binary_pairs())
    def test_binary_sumo_relations(self, compiler, cnl, expected):
        assert compiler.compile(cnl) == expected