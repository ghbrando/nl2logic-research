import pytest


# ── Tests ──────────────────────────────────────────────────────────────────

class TestUniversalQuantifier:

    def test_every_bare(self, compiler):
        """every ?x is-a C with no implies — bare forall"""
        assert compiler.compile(
            "every ?x is-a MilitaryUnit"
        ) == "(forall (?x) (instance ?x MilitaryUnit))"

    def test_every_with_implies(self, compiler):
        assert compiler.compile(
            "every ?x is-a Process implies agent ?x ?y"
        ) == "(forall (?x) (=> (instance ?x Process) (agent ?x ?y)))"

    def test_every_with_binary_consequent(self, compiler):
        assert compiler.compile(
            "every ?x is-a Soldier implies member ?x ?unit"
        ) == "(forall (?x) (=> (instance ?x Soldier) (member ?x ?unit)))"

    def test_every_with_instance_consequent(self, compiler):
        assert compiler.compile(
            "every ?x is-a MilitaryProcess implies ?x is-a Process"
        ) == "(forall (?x) (=> (instance ?x MilitaryProcess) (instance ?x Process)))"


class TestExistentialQuantifier:

    def test_some_basic(self, compiler):
        assert compiler.compile(
            "some ?x is-a Soldier"
        ) == "(exists (?x) (instance ?x Soldier))"

    def test_some_military_process(self, compiler):
        assert compiler.compile(
            "some ?x is-a MilitaryProcess"
        ) == "(exists (?x) (instance ?x MilitaryProcess))"

    def test_some_named_var(self, compiler):
        assert compiler.compile(
            "some ?unit is-a MilitaryUnit"
        ) == "(exists (?unit) (instance ?unit MilitaryUnit))"


class TestNegativeUniversalQuantifier:

    def test_no_bare(self, compiler):
        """no ?x is-a C with no implies — forall not"""
        assert compiler.compile(
            "no ?x is-a Weapon"
        ) == "(forall (?x) (not (instance ?x Weapon)))"

    def test_no_with_implies(self, compiler):
        assert compiler.compile(
            "no ?x is-a Weapon implies member ?x ?unit"
        ) == "(forall (?x) (=> (instance ?x Weapon) (not (member ?x ?unit))))"


class TestQuantifierVariableScoping:

    def test_variable_name_preserved_in_body(self, compiler):
        """The bound variable must appear in the body, not be renamed."""
        kif = compiler.compile("every ?soldier is-a Soldier implies agent ?mission ?soldier")
        assert "?soldier" in kif
        assert kif == "(forall (?soldier) (=> (instance ?soldier Soldier) (agent ?mission ?soldier)))"

    def test_free_variable_preserved(self, compiler):
        """Variables not bound by the quantifier must pass through unchanged."""
        kif = compiler.compile("every ?x is-a Process implies agent ?x ?y")
        assert "?y" in kif