import json
import pytest
from pathlib import Path

SUMO_RELATIONS = Path("data/training_pairs/sumo_relations.jsonl")


def _load_relations() -> list[dict]:
    with open(SUMO_RELATIONS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _nary_pairs() -> list[tuple[str, str]]:
    pairs = []
    for r in _load_relations():
        arity = r.get("arity")
        if arity and arity >= 3:
            term = r["term"]
            vars_ = [f"?a{i}" for i in range(1, arity + 1)]
            cnl = f"{term} [ {' , '.join(vars_)} ]"
            kif = f"({term} {' '.join(vars_)})"
            pairs.append((cnl, kif))
        if len(pairs) >= 20:
            break
    return pairs


def _nary_relations() -> list[dict]:
    return [r for r in _load_relations() if r.get("arity") and r["arity"] >= 3]


# ── Tests ──────────────────────────────────────────────────────────────────

class TestNaryRelations:

    # Arity 3 — hand-written spot checks
    def test_ternary_basic(self, compiler):
        assert compiler.compile("between [ ?a , ?b , ?c ]") == "(between ?a ?b ?c)"

    def test_ternary_located(self, compiler):
        assert compiler.compile("located [ ?obj , ?region , ?time ]") == "(located ?obj ?region ?time)"

    # Arity 4
    def test_quaternary_transfers_ownership(self, compiler):
        assert compiler.compile(
            "transfersOwnership [ ?obj , ?from , ?to , ?proc ]"
        ) == "(transfersOwnership ?obj ?from ?to ?proc)"

    # Argument count in output must match declared SUMO arity
    @pytest.mark.parametrize("relation", _nary_relations())
    def test_arity_matches_sumo_schema(self, compiler, relation):
        term  = relation["term"]
        arity = relation["arity"]
        vars_ = [f"?a{i}" for i in range(1, arity + 1)]
        cnl   = f"{term} [ {' , '.join(vars_)} ]"
        kif   = compiler.compile(cnl)
        # KIF form: (relName arg1 arg2 ... argN)
        tokens = kif.strip("()").split()
        args   = tokens[1:]  # drop relation name
        assert len(args) == arity, (
            f"{term}: expected arity {arity}, got {len(args)} in '{kif}'"
        )

    # Variable names must be preserved positionally
    def test_variable_names_preserved(self, compiler):
        cnl = "between [ ?alpha , ?bravo , ?charlie ]"
        kif = compiler.compile(cnl)
        assert kif == "(between ?alpha ?bravo ?charlie)"

    # Binary syntax must NOT be accepted for n-ary relations
    def test_binary_syntax_rejected_for_ternary(self, compiler):
        with pytest.raises(Exception):
            compiler.compile("between ?a ?b")  # missing third arg

    # SUMO-derived parametrized coverage
    @pytest.mark.parametrize("cnl,expected", _nary_pairs())
    def test_nary_sumo_relations(self, compiler, cnl, expected):
        assert compiler.compile(cnl) == expected