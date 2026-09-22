"""Strict relation validation distinguishes class symbols from individuals."""

import pytest

from src.compiler.compiler import CNLCompiler


@pytest.fixture(scope="module")
def compiler():
    return CNLCompiler()


def test_class_level_relation_accepts_class_arguments(compiler):
    assert compiler.compile("partTypes Object Object", require_argument_kinds=True) == (
        "(partTypes Object Object)"
    )


def test_instance_relation_rejects_class_arguments(compiler):
    with pytest.raises(ValueError, match="expects an instance"):
        compiler.compile("agent Process AutonomousAgent", require_argument_kinds=True)


def test_class_level_relation_rejects_individual_variable(compiler):
    with pytest.raises(ValueError, match="expects a class"):
        compiler.compile("partTypes ?x Object", require_argument_kinds=True)


def test_mixed_relation_checks_each_position(compiler):
    assert compiler.compile("capability [ Process, ?role, ?object ]", require_argument_kinds=True) == (
        "(capability Process ?role ?object)"
    )
    with pytest.raises(ValueError, match="argument 1 expects a class"):
        compiler.compile("capability [ ?process, ?role, ?object ]", require_argument_kinds=True)
    with pytest.raises(ValueError, match="argument 2 expects an instance"):
        compiler.compile("capability [ Process, CaseRole, ?object ]", require_argument_kinds=True)


def test_closed_validation_still_rejects_free_individuals(compiler):
    with pytest.raises(ValueError, match="Unbound variables"):
        compiler.compile(
            "agent ?process ?agent", require_closed=True, require_argument_kinds=True
        )
