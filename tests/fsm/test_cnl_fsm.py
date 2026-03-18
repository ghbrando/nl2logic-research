"""
Tests for src/fsm/cnl_fsm.py ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â Stage 2 FSM integration.

Test strategy
-------------
- Grammar building and LALR(1) validation use only Lark (no model needed).
- validate_output() delegates to CNLCompiler (no model needed).
- CNLSampler construction is tested with MagicMock (no model needed).
- CNLSampler.sample() is tested with a MagicMock model that returns a
  known CNL string; the custom Hugging Face constrained integration itself is tested end-to-end
  only in integration tests (not here).

All tests use SUMO terms known to exist in sumo_classes.jsonl /
sumo_relations.jsonl so they do not depend on SUMO vocab changes.
"""

import math
import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput
from unittest.mock import MagicMock

from src.fsm.cnl_fsm import (
    CNLSampler,
    UnsupportedInputError,
    build_vocabulary_grammar,
    build_xgrammar_grammar,
    validate_base_grammar_lalr,
)


# ---------------------------------------------------------------------------
# Grammar building
# ---------------------------------------------------------------------------

class TestBuildVocabularyGrammar:

    def test_returns_string(self, grammar_str):
        assert isinstance(grammar_str, str)
        assert len(grammar_str) > 100

    def test_contains_class_term_terminal(self, grammar_str):
        assert "CLASS_TERM" in grammar_str

    def test_contains_relation_terminal(self, grammar_str):
        assert "RELATION" in grammar_str

    def test_known_class_in_grammar(self, grammar_str):
        # 'Process' is a core SUMO class present in sumo_classes.jsonl
        assert "Process" in grammar_str

    def test_known_relation_in_grammar(self, grammar_str):
        # 'agent' is a core SUMO relation present in sumo_relations.jsonl
        assert "agent" in grammar_str

    def test_open_class_regex_replaced(self, grammar_str):
        # The open-ended regex must be gone
        assert "/[A-Z][a-zA-Z0-9_]*/" not in grammar_str

    def test_open_relation_regex_replaced(self, grammar_str):
        assert "/[a-z][a-zA-Z0-9]*/" not in grammar_str


class TestBuildXGrammarGrammar:

    def test_returns_string(self, constrained_grammar_str):
        assert isinstance(constrained_grammar_str, str)
        assert len(constrained_grammar_str) > 100

    def test_uses_gbnf_rule_syntax(self, constrained_grammar_str):
        assert "root ::= sentence" in constrained_grammar_str
        assert "class-term ::=" in constrained_grammar_str
        assert "relation ::=" in constrained_grammar_str

    def test_omits_lark_only_constructs(self, constrained_grammar_str):
        assert "%ignore" not in constrained_grammar_str
        assert "//" not in constrained_grammar_str
        assert "CLASS_TERM  :" not in constrained_grammar_str
        assert "RELATION    :" not in constrained_grammar_str

    def test_known_vocab_terms_present(self, constrained_grammar_str):
        assert '"Process"' in constrained_grammar_str
        assert '"agent"' in constrained_grammar_str

    def test_builder_function_matches_fixture(self, constrained_grammar_str):
        assert constrained_grammar_str == build_xgrammar_grammar()

    def test_xgrammar_compiles_when_available(self, constrained_grammar_str):
        xgrammar = pytest.importorskip("xgrammar")
        xgrammar.Grammar.from_ebnf(constrained_grammar_str)


# ---------------------------------------------------------------------------
# LALR(1) compatibility
# ---------------------------------------------------------------------------

class TestGrammarLalr1:

    def test_base_grammar_is_lalr1(self):
        """The structural base grammar (open terminals) must be LALR(1)-compatible.

        The vocabulary-closed grammar uses Earley at parse time because a
        29,649-term alternation makes LALR table construction impractical.
        """
        validate_base_grammar_lalr()  # raises RuntimeError if not LALR(1)


# ---------------------------------------------------------------------------
# Grammar accepts valid CNL
# ---------------------------------------------------------------------------

class TestGrammarAcceptsValidCNL:

    def test_instance_assertion(self, vocab_parser):
        vocab_parser.parse("?x is-a Process")

    def test_instance_named_var(self, vocab_parser):
        vocab_parser.parse("?attack is-a MilitaryProcess")

    def test_subclass_assertion(self, vocab_parser):
        vocab_parser.parse("Process subclass-of Entity")

    def test_binary_assertion(self, vocab_parser):
        vocab_parser.parse("agent ?x Process")

    def test_nary_assertion(self, vocab_parser):
        vocab_parser.parse("between [ ?a , ?b , ?c ]")

    def test_universal_quantifier(self, vocab_parser):
        vocab_parser.parse("every ?x is-a Process")

    def test_universal_with_implies(self, vocab_parser):
        vocab_parser.parse("every ?x is-a Process implies ?x is-a Entity")

    def test_existential_quantifier(self, vocab_parser):
        vocab_parser.parse("some ?x is-a Process")

    def test_negative_quantifier(self, vocab_parser):
        vocab_parser.parse("no ?x is-a Process")

    def test_negated_atomic_assertion(self, vocab_parser):
        vocab_parser.parse("not ?x is-a Process")

    def test_conditional(self, vocab_parser):
        vocab_parser.parse("if ?x is-a Process then ?x is-a Entity")


# ---------------------------------------------------------------------------
# Grammar rejects invalid CNL
# ---------------------------------------------------------------------------

class TestGrammarRejectsInvalidCNL:

    def test_unknown_class_rejected(self, vocab_parser):
        with pytest.raises(UnexpectedInput):
            vocab_parser.parse("?x is-a FakeClass999NotInSUMO")

    def test_unknown_relation_rejected(self, vocab_parser):
        with pytest.raises((UnexpectedInput, Exception)):
            vocab_parser.parse("fakeRelXYZ ?x ?y")

    def test_bare_word_rejected(self, vocab_parser):
        with pytest.raises(UnexpectedInput):
            vocab_parser.parse("hello world")

    def test_empty_string_rejected(self, vocab_parser):
        with pytest.raises(UnexpectedInput):
            vocab_parser.parse("")

    def test_kif_syntax_rejected(self, vocab_parser):
        with pytest.raises(UnexpectedInput):
            vocab_parser.parse("(instance ?x Process)")


# ---------------------------------------------------------------------------
# validate_output (no model needed)
# ---------------------------------------------------------------------------

class TestValidateOutput:

    def test_instance_returns_kif(self, sampler):
        assert sampler.validate_output("?x is-a Process") == "(instance ?x Process)"

    def test_subclass_returns_kif(self, sampler):
        assert sampler.validate_output("Process subclass-of Entity") == "(subclass Process Entity)"

    def test_binary_returns_kif(self, sampler):
        assert sampler.validate_output("agent ?x Process") == "(agent ?x Process)"

    def test_nary_returns_kif(self, sampler):
        assert sampler.validate_output("between [ ?a , ?b , ?c ]") == "(between ?a ?b ?c)"

    def test_negated_assertion_returns_kif(self, sampler):
        assert sampler.validate_output("not agent ?x Process") == "(not (agent ?x Process))"

    def test_invalid_cnl_raises(self, sampler):
        with pytest.raises(Exception):
            sampler.validate_output("this is not cnl at all !!!!")

    def test_unknown_class_raises(self, sampler):
        with pytest.raises(Exception):
            sampler.validate_output("?x is-a NotARealSUMOClass99999")


# ---------------------------------------------------------------------------
# CNLSampler construction
# ---------------------------------------------------------------------------

class TestCNLSamplerConstruction:

    def test_construction_with_mock(self, grammar_str):
        s = CNLSampler(MagicMock(), MagicMock(), grammar_str=grammar_str)
        assert s is not None

    def test_construction_builds_grammar_when_not_provided(self):
        # Does not crash; grammar is built from SUMO vocab files
        s = CNLSampler(MagicMock(), MagicMock())
        assert s is not None

    def test_validate_output_available_without_backend(self, sampler):
        # validate_output must work even though constrained decoding is not exercised here
        result = sampler.validate_output("?x is-a Process")
        assert result == "(instance ?x Process)"


# ---------------------------------------------------------------------------
# Abstain heuristic
# ---------------------------------------------------------------------------

class TestCNLSamplerAbstain:

    def test_single_short_sentence_does_not_abstain(self, sampler):
        assert sampler.abstain_if_unsupported("Every Process is in the agent relation with an Agent.") is False

    def test_instruction_prefix_is_ignored(self, sampler):
        assert sampler.abstain_if_unsupported("translate to CNL: Something is an instance of Process.") is False

    def test_multiple_sentences_abstain(self, sampler):
        assert sampler.abstain_if_unsupported("Process is an Entity. Agent is an Entity.") is True

    def test_long_input_abstains(self, sampler):
        nl = " ".join(f"token{i}" for i in range(70))
        assert sampler.abstain_if_unsupported(nl) is True

    def test_coreference_marker_abstains(self, sampler):
        assert sampler.abstain_if_unsupported("If it advances, the unit attacks.") is True

    @pytest.mark.parametrize(
        "nl",
        [
            "Every battle can have a military unit as its patient.",
            "Every transportation process has a region as its destination.",
            "Every transportation process has a region as its origin.",
            "A process can have an agent as its agent.",
        ],
    )
    def test_supported_possessive_relation_phrase_does_not_abstain(self, sampler, nl):
        assert sampler.abstain_if_unsupported(nl) is False

    def test_it_is_not_the_case_wrapper_does_not_trigger_discourse_abstain(self, sampler):
        nl = "It is not the case that the agent relation holds between a MilitaryProcess and an AutonomousAgent."
        assert sampler.abstain_if_unsupported(nl) is False

    def test_other_its_usage_still_abstains(self, sampler):
        nl = "If its commander retreats, the unit halts."
        assert sampler.abstain_if_unsupported(nl) is True

    def test_sample_logs_warning_and_abstains(self, sampler, caplog):
        with caplog.at_level("WARNING"):
            with pytest.raises(UnsupportedInputError, match="multiple sentences detected"):
                sampler.sample("Process is an Entity. Agent is an Entity.")

        sampler._hf_model.generate.assert_not_called()
        assert "Abstaining from CNL generation for unsupported input" in caplog.text


# ---------------------------------------------------------------------------
# CNLSampler.sample() ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â mock model, no real constrained runtime needed
# ---------------------------------------------------------------------------

class TestCNLSamplerSample:

    def test_sample_calls_hf_generate_with_prefix_constraint(self, constrained_grammar_str):
        class SampleTokenizer:
            def __init__(self):
                self.prompts_seen = []
                self._ids = {}

            def __call__(self, texts=None, **kwargs):
                assert texts is not None
                if kwargs.get("return_tensors") == "pt":
                    self.prompts_seen.extend(texts)
                    return {
                        "input_ids": FakeTensorBatch([[101, 102]]),
                        "attention_mask": FakeTensorBatch([[1, 1]]),
                    }

                assert kwargs.get("add_special_tokens") is False
                if texts not in self._ids:
                    self._ids[texts] = len(self._ids) + 1000
                return {"input_ids": [self._ids[texts]]}

            def batch_decode(self, sequences, skip_special_tokens=True):
                assert skip_special_tokens is True
                return ["?x is-a Process"]

        class SampleModel:
            def __init__(self):
                self.device = "cpu"
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [[201, 202, 203]]

        model = SampleModel()
        tokenizer = SampleTokenizer()
        sampler = CNLSampler(model, tokenizer, grammar_str=constrained_grammar_str)
        fake_constraint = object()
        sampler._get_prefix_constraint = lambda prompt: fake_constraint

        result = sampler.sample("translate: every soldier is a combatant")

        assert result == "?x is-a Process"
        assert tokenizer.prompts_seen == ["translate: every soldier is a combatant"]
        assert model.calls[0]["max_new_tokens"] == 200
        assert model.calls[0]["prefix_allowed_tokens_fn"] is fake_constraint
        assert model.calls[0]["num_beams"] == 4
        assert model.calls[0]["length_penalty"] == pytest.approx(1.15)
        assert model.calls[0]["early_stopping"] is True
        assert model.calls[0]["renormalize_logits"] is True
        assert model.calls[0]["input_ids"].moved_to == "cpu"

class FakeTensorBatch:
    def __init__(self, values):
        self.values = values
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.prompts_seen: list[str] = []
        self._ids: dict[str, int] = {}

    def __call__(self, texts=None, **kwargs):
        assert texts is not None
        if kwargs.get("return_tensors") == "pt":
            self.prompts_seen.extend(texts)
            return {
                "input_ids": FakeTensorBatch([[101, 102]]),
                "attention_mask": FakeTensorBatch([[1, 1]]),
            }

        assert kwargs.get("add_special_tokens") is False
        if texts not in self._ids:
            self._ids[texts] = len(self._ids) + 1000
        return {"input_ids": [self._ids[texts]]}

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ["?x is-a Process"]


class FakeGenerationOutput:
    def __init__(self, *, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeConfidenceModel:
    def __init__(self, scores):
        self.device = "cuda:0"
        self._scores = scores
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGenerationOutput(
            sequences=[[201, 202, 203]],
            scores=self._scores,
        )


class TestCNLSamplerConfidence:

    def test_sample_with_confidence_returns_mean_top1_probability(self, grammar_str):
        scores = [
            [[0.0, 0.0]],
            [[math.log(3.0), 0.0]],
            [[math.log(9.0), 0.0]],
        ]
        model = FakeConfidenceModel(scores)
        tokenizer = FakeTokenizer()
        sampler = CNLSampler(model, tokenizer, grammar_str=grammar_str)

        cnl, confidence = sampler.sample_with_confidence("translate to CNL: Something is an instance of Process.")

        assert cnl == "?x is-a Process"
        assert confidence == pytest.approx((0.5 + 0.75 + 0.9) / 3)
        assert tokenizer.prompts_seen == ["translate to CNL: Something is an instance of Process."]
        assert model.calls[0]["input_ids"].moved_to == "cuda:0"
        assert model.calls[0]["max_new_tokens"] == 200
        assert model.calls[0]["return_dict_in_generate"] is True
        assert model.calls[0]["output_scores"] is True

    def test_sample_with_confidence_logs_low_confidence_warning(self, grammar_str, caplog):
        model = FakeConfidenceModel(scores=[[[0.0, 0.0]]])
        tokenizer = FakeTokenizer()
        sampler = CNLSampler(
            model,
            tokenizer,
            grammar_str=grammar_str,
            confidence_threshold=0.7,
        )

        with caplog.at_level("WARNING"):
            cnl, confidence = sampler.sample_with_confidence("Something is an instance of Process.")

        assert cnl == "?x is-a Process"
        assert confidence == pytest.approx(0.5)
        assert "Low-confidence CNL generation" in caplog.text







class PrefixConstraintTokenizer:
    def __init__(self):
        self._ids: dict[str, int] = {}
        self.eos_token_id = 0

    def __call__(self, text=None, **kwargs):
        if text is None:
            raise ValueError("text is required")
        if text not in self._ids:
            self._ids[text] = len(self._ids) + 1
        return {"input_ids": [self._ids[text]]}


class TestPrefixConstraintStructure:

    def test_class_slot_does_not_leak_into_other_template_continuations(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Process", "Entity"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint()
        prefix = tokenizer("?x is-a", add_special_tokens=False)["input_ids"]
        prefix += tokenizer(" Process", add_special_tokens=False)["input_ids"]

        allowed = constraint(0, prefix)

        assert allowed == [tokenizer.eos_token_id]
        assert tokenizer(" implies", add_special_tokens=False)["input_ids"][0] not in allowed
    def test_accepting_prefix_ending_in_eos_stays_on_eos(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Process", "Entity"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint()
        prefix = tokenizer("?x is-a", add_special_tokens=False)["input_ids"]
        prefix += tokenizer(" Process", add_special_tokens=False)["input_ids"]
        prefix.append(tokenizer.eos_token_id)

        allowed = constraint(0, prefix)

        assert allowed == [tokenizer.eos_token_id]
class TestPromptAwareConstraints:

    def test_infers_binary_agent_plan_from_prompt(self, monkeypatch):
        sampler = CNLSampler(MagicMock(), MagicMock())

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["MilitaryProcess", "AutonomousAgent", "Process"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent", "destination", "origin", "located"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        plan = sampler._infer_prompt_plan("A military process has an autonomous agent.")

        assert plan is not None
        assert plan.template == "binary"
        assert plan.relation_terms == ("agent",)
        assert plan.first_terms == ("MilitaryProcess",)
        assert plan.second_terms == ("AutonomousAgent",)

    def test_infers_conditional_destination_plan_from_prompt(self, monkeypatch):
        sampler = CNLSampler(MagicMock(), MagicMock())

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Transportation", "Region", "Process"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["destination", "origin"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        plan = sampler._infer_prompt_plan("Every transportation process has a region as its destination.")

        assert plan is not None
        assert plan.template == "conditional_every"
        assert plan.relation_terms == ("destination",)
        assert plan.first_terms == ("Transportation",)
        assert plan.second_terms == ("Region",)

    def test_infers_nary_plan_from_prompt(self, monkeypatch):
        sampler = CNLSampler(MagicMock(), MagicMock())

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Area", "MilitaryUnit", "Region"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["between"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        plan = sampler._infer_prompt_plan("The between relation holds among an Area, a MilitaryUnit, and a Region.")

        assert plan is not None
        assert plan.template == "nary"
        assert plan.relation_terms == ("between",)
        assert plan.first_terms == ("Area",)
        assert plan.second_terms == ("MilitaryUnit",)
        assert plan.third_terms == ("Region",)

    def test_infers_negated_nary_plan_from_prompt(self, monkeypatch):
        sampler = CNLSampler(MagicMock(), MagicMock())

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Area", "MilitaryUnit", "Region"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["between"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        plan = sampler._infer_prompt_plan("Area, MilitaryUnit, and Region do not stand in the between relation.")

        assert plan is not None
        assert plan.template == "negation_nary"
        assert plan.relation_terms == ("between",)
        assert plan.first_terms == ("Area",)
        assert plan.second_terms == ("MilitaryUnit",)
        assert plan.third_terms == ("Region",)

    def test_prompt_specific_binary_constraint_disallows_instance_start(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["MilitaryProcess", "AutonomousAgent", "Process"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint("A military process has an autonomous agent.")
        allowed = constraint(0, [])

        assert tokenizer("agent", add_special_tokens=False)["input_ids"][0] in allowed
        assert tokenizer("?x is-a", add_special_tokens=False)["input_ids"][0] not in allowed

    def test_prompt_specific_nary_constraint_disallows_binary_start(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Area", "MilitaryUnit", "Region"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["between", "agent"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint("The between relation holds among an Area, a MilitaryUnit, and a Region.")
        allowed = constraint(0, [])

        assert tokenizer("between", add_special_tokens=False)["input_ids"][0] in allowed
        assert tokenizer("agent", add_special_tokens=False)["input_ids"][0] not in allowed
    def test_unmatched_prompts_build_prompt_specific_constraints(self, monkeypatch):
        sampler = CNLSampler(MagicMock(), MagicMock())
        built_constraints: dict[str | None, object] = {}
        build_calls: list[str | None] = []

        monkeypatch.setattr(sampler, "_control_literal", lambda: None)

        def fake_build(prompt=None):
            build_calls.append(prompt)
            constraint = object()
            built_constraints[prompt] = constraint
            return constraint

        monkeypatch.setattr(sampler, "_build_prefix_constraint", fake_build)

        first = sampler._get_prefix_constraint("Completely unmatched prompt one")
        second = sampler._get_prefix_constraint("Different unmatched prompt two")
        repeated = sampler._get_prefix_constraint("Completely unmatched prompt one")

        assert first is built_constraints["Completely unmatched prompt one"]
        assert second is built_constraints["Different unmatched prompt two"]
        assert repeated is first
        assert build_calls == ["Completely unmatched prompt one", "Different unmatched prompt two"]

    def test_unmatched_prompt_without_structure_allows_only_eos(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Army", "Planning", "Process"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent", "located"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint("Army operations support planning.")
        allowed = constraint(0, [])

        assert allowed == [tokenizer.eos_token_id]
        assert tokenizer("?x is-a", add_special_tokens=False)["input_ids"][0] not in allowed
        assert tokenizer("Army", add_special_tokens=False)["input_ids"][0] not in allowed

    def test_unmatched_prompt_with_relation_signal_narrows_relation_start(self, monkeypatch):
        tokenizer = PrefixConstraintTokenizer()
        model = MagicMock()
        model.config.decoder_start_token_id = None
        sampler = CNLSampler(model, tokenizer)

        def fake_load_terms(path, key="term"):
            path_str = str(path)
            if path_str.endswith("sumo_classes.jsonl"):
                return ["Army", "Planning", "Process"]
            if path_str.endswith("sumo_relations.jsonl"):
                return ["agent", "located"]
            raise AssertionError(f"Unexpected load path: {path}")

        monkeypatch.setattr("src.fsm.cnl_fsm._load_terms", fake_load_terms)

        constraint = sampler._build_prefix_constraint("Army formations require agent coordination.")
        allowed = constraint(0, [])

        assert tokenizer("agent", add_special_tokens=False)["input_ids"][0] in allowed
        assert tokenizer("located", add_special_tokens=False)["input_ids"][0] not in allowed
        assert tokenizer("?x is-a", add_special_tokens=False)["input_ids"][0] not in allowed

    def test_sample_abstains_when_constraint_has_no_supported_start(self):
        model = MagicMock()
        model.device = "cpu"
        tokenizer = FakeTokenizer()
        sampler = CNLSampler(model, tokenizer)

        class EmptyConstraint:
            _eos_token_id = 0

            def __call__(self, _batch_id, input_ids):
                return [0]

        sampler._get_prefix_constraint = lambda prompt: EmptyConstraint()

        with pytest.raises(UnsupportedInputError, match="no lexically supported constrained continuation"):
            sampler.sample("Army operations support planning.")

        model.generate.assert_not_called()
