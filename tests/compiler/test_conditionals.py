import pytest


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSingleAntecedent:

    def test_basic_implication(self, compiler):
        assert compiler.compile(
            "if agent ?attack ?s then patient ?attack ?t"
        ) == "(=> (agent ?attack ?s) (patient ?attack ?t))"

    def test_implication_with_member(self, compiler):
        assert compiler.compile(
            "if member ?soldier ?unit then agent ?mission ?soldier"
        ) == "(=> (member ?soldier ?unit) (agent ?mission ?soldier))"

    def test_implication_with_instance_consequent(self, compiler):
        assert compiler.compile(
            "if agent ?a ?s then ?a is-a MilitaryProcess"
        ) == "(=> (agent ?a ?s) (instance ?a MilitaryProcess))"

    def test_implication_with_instance_antecedent(self, compiler):
        assert compiler.compile(
            "if ?x is-a Weapon then instrument ?attack ?x"
        ) == "(=> (instance ?x Weapon) (instrument ?attack ?x))"


class TestConjunctiveAntecedent:

    def test_two_conjuncts(self, compiler):
        assert compiler.compile(
            "if agent ?a ?s and instrument ?a ?w then patient ?a ?t"
        ) == "(=> (and (agent ?a ?s) (instrument ?a ?w)) (patient ?a ?t))"

    def test_three_conjuncts(self, compiler):
        assert compiler.compile(
            "if agent ?a ?s and instrument ?a ?w and origin ?a ?base then destination ?a ?obj"
        ) == "(=> (and (agent ?a ?s) (instrument ?a ?w) (origin ?a ?base)) (destination ?a ?obj))"

    def test_conjunct_with_instance(self, compiler):
        assert compiler.compile(
            "if ?x is-a Soldier and member ?x ?unit then agent ?mission ?x"
        ) == "(=> (and (instance ?x Soldier) (member ?x ?unit)) (agent ?mission ?x))"


class TestConditionalVariablePreservation:

    def test_all_variables_preserved(self, compiler):
        """Every variable in the CNL must appear in the KIF output unchanged."""
        cnl = "if agent ?attack ?soldier and instrument ?attack ?weapon then patient ?attack ?target"
        kif = compiler.compile(cnl)
        for var in ["?attack", "?soldier", "?weapon", "?target"]:
            assert var in kif, f"Variable {var} missing from output: {kif}"

    def test_shared_variable_across_antecedent_and_consequent(self, compiler):
        """A variable used in both antecedent and consequent must be the same token."""
        kif = compiler.compile("if agent ?a ?s then patient ?a ?t")
        # ?a must appear exactly twice — once in antecedent, once in consequent
        assert kif.count("?a") == 2