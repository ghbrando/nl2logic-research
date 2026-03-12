from unittest.mock import MagicMock

import pytest

from src.fsm.cnl_fsm import CNLSampler, build_vocabulary_grammar


@pytest.fixture(scope="session")
def grammar_str() -> str:
    """Build the vocabulary-closed grammar once per test session."""
    return build_vocabulary_grammar()


@pytest.fixture(scope="session")
def vocab_parser(grammar_str: str):
    """Lark Earley parser built from the vocabulary grammar.

    The vocabulary-closed grammar has ~29,649-term alternations which make
    LALR table construction impractical.  Earley handles it efficiently and
    is what Outlines uses internally for CFG-constrained generation.
    """
    from lark import Lark
    return Lark(grammar_str, parser="earley", ambiguity="resolve")


@pytest.fixture(scope="session")
def sampler(grammar_str: str) -> CNLSampler:
    """CNLSampler with a mock HF model and tokenizer."""
    return CNLSampler(MagicMock(), MagicMock(), grammar_str=grammar_str)
