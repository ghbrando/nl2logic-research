import pytest


class TestAtomicNegation:

    def test_negated_instance_assertion(self, compiler):
        assert compiler.compile(
            "not ?x is-a Process"
        ) == "(not (instance ?x Process))"

    def test_negated_binary_assertion(self, compiler):
        assert compiler.compile(
            "not agent ?attack ?soldier"
        ) == "(not (agent ?attack ?soldier))"

    def test_negated_nary_assertion(self, compiler):
        assert compiler.compile(
            "not between [ ?a , ?b , ?c ]"
        ) == "(not (between ?a ?b ?c))"

    def test_negation_in_conditional_antecedent(self, compiler):
        assert compiler.compile(
            "if not agent ?attack ?soldier then ?attack is-a MilitaryProcess"
        ) == "(=> (not (agent ?attack ?soldier)) (instance ?attack MilitaryProcess))"

    def test_negation_as_quantified_consequent(self, compiler):
        assert compiler.compile(
            "every ?x is-a Process implies not agent ?x ?y"
        ) == "(forall (?x) (=> (instance ?x Process) (not (agent ?x ?y))))"
