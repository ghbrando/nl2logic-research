"""
Tests for src/data/generate_pairs.py

All tests use a small in-memory compiler fixture so they are fast and
do not depend on the full 29,649-term SUMO vocab being available in a
specific location.  SUMO-specific terms used in fixtures (Process, Entity,
agent) are real vocab terms present in sumo_classes.jsonl / sumo_relations.jsonl.
"""

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.compiler.compiler import CNLCompiler
from src.data.generate_pairs import (
    _article_for,
    _clean_doc,
    count_pairs_by_pattern,
    gen_binary_pairs,
    gen_conditional_every_pairs,
    gen_conditional_if_pairs,
    gen_existential_pairs,
    gen_instance_pairs,
    gen_negation_pairs,
    gen_nary_pairs,
    gen_subclass_pairs,
    load_classes,
    load_relations,
    pascal_to_natural,
    select_output_pairs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compiler():
    return CNLCompiler()


# Minimal class/relation fixtures using real SUMO terms
SAMPLE_CLASSES = [
    {"term": "Process"},
    {"term": "Entity", "doc": "The broadest possible kind of entity."},
    {"term": "AutonomousAgent"},
]

SAMPLE_RELATIONS = [
    {
        "term": "agent",
        "arity": 2,
        "signature": {"1": "Process", "2": "AutonomousAgent"},
        "doc": "agent relates a Process to an AutonomousAgent.",
    },
    {
        "term": "instance",   # intentionally invalid RELATION (starts uppercase? no — 'instance' lowercase but is a SUMO keyword)
        "arity": 2,
        "signature": {"1": "Entity", "2": "SetOrClass"},
    },
]

SAMPLE_NARY_RELATION = {
    "term": "between",
    "arity": 3,
    "signature": {"1": "Object", "2": "Object", "3": "Object"},
    "doc": "between places one Object between two other Objects.",
}

SAMPLE_NEGATION_BINARY_RELATION = {
    "term": "located",
    "arity": 2,
    "signature": {"1": "Object", "2": "Object"},
    "doc": "located places one Object at another Object.",
}

# Only 'agent' passes the compiler's relation validator; 'instance' does too
# (it is in sumo_relations.jsonl)


# ---------------------------------------------------------------------------
# _clean_doc
# ---------------------------------------------------------------------------

class TestCleanDoc:

    def test_strips_sumo_refs(self):
        doc = "&%Process is a kind of &%Entity."
        assert "&%" not in _clean_doc(doc)

    def test_keeps_term_text(self):
        doc = "&%Process is a kind of &%Entity."
        cleaned = _clean_doc(doc)
        assert "Process" in cleaned
        assert "Entity" in cleaned

    def test_truncates_to_first_sentence(self):
        doc = "First sentence here. Second sentence here. Third."
        result = _clean_doc(doc)
        assert result.endswith("First sentence here.")
        assert "Second" not in result

    def test_returns_empty_for_short_input(self):
        assert _clean_doc("Hi.") == ""
        assert _clean_doc("") == ""

    def test_no_crash_on_no_period(self):
        doc = "A" * 250  # long string, no period
        result = _clean_doc(doc)
        assert len(result) <= 300


# ---------------------------------------------------------------------------
# pascal_to_natural / _article_for
# ---------------------------------------------------------------------------

class TestPascalToNatural:

    def test_splits_pascal_case(self):
        assert pascal_to_natural("MilitaryProcess") == "military process"

    def test_splits_multiple_words(self):
        assert pascal_to_natural("AutonomousAgent") == "autonomous agent"

    def test_all_caps_stays_together(self):
        assert pascal_to_natural("AAM") == "aam"

    def test_lowercase_passthrough(self):
        assert pascal_to_natural("agent") == "agent"

    def test_single_word_uppercase(self):
        assert pascal_to_natural("Process") == "process"

    def test_mixed_case_acronym_prefix(self):
        assert pascal_to_natural("XMLParser") == "xml parser"


class TestArticleFor:

    def test_consonant_gets_a(self):
        assert _article_for("process") == "a"

    def test_vowel_gets_an(self):
        assert _article_for("entity") == "an"

    def test_uppercase_vowel_gets_an(self):
        assert _article_for("Entity") == "an"

    def test_empty_string_gets_a(self):
        assert _article_for("") == "a"


# ---------------------------------------------------------------------------
# load_classes / load_relations
# ---------------------------------------------------------------------------

class TestLoadClasses:

    def test_filters_hyphenated_terms(self, tmp_path):
        data = [
            {"term": "ValidClass"},
            {"term": "Bad-Class"},    # hyphen — invalid CLASS_TERM
            {"term": "also_valid"},   # starts lowercase — not a class
            {"term": "AlsoValid"},
        ]
        p = tmp_path / "classes.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
        result = load_classes(p)
        terms = [r["term"] for r in result]
        assert "ValidClass" in terms
        assert "AlsoValid" in terms
        assert "Bad-Class" not in terms
        assert "also_valid" not in terms

    def test_preserves_doc_field(self, tmp_path):
        data = [{"term": "Process", "doc": "A doc string."}]
        p = tmp_path / "classes.jsonl"
        p.write_text(json.dumps(data[0]), encoding="utf-8")
        result = load_classes(p)
        assert result[0]["doc"] == "A doc string."


class TestLoadRelations:

    def test_filters_invalid_terms(self, tmp_path):
        data = [
            {"term": "agent", "arity": 2, "signature": {}},
            {"term": "bad_rel", "arity": 2, "signature": {}},   # underscore
            {"term": "BadRel", "arity": 2, "signature": {}},    # uppercase
            {"term": "noarity", "signature": {}},               # no arity
        ]
        p = tmp_path / "rels.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
        result = load_relations(p)
        terms = [r["term"] for r in result]
        assert "agent" in terms
        assert "bad_rel" not in terms
        assert "BadRel" not in terms
        assert "noarity" not in terms


# ---------------------------------------------------------------------------
# gen_instance_pairs
# ---------------------------------------------------------------------------

class TestGenInstancePairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_instance_pairs(SAMPLE_CLASSES, compiler)
        assert len(pairs) > 0

    def test_record_shape(self, compiler):
        pairs = gen_instance_pairs(SAMPLE_CLASSES, compiler)
        for p in pairs:
            assert set(p.keys()) == {"nl", "cnl", "kif", "pattern"}

    def test_pattern_label(self, compiler):
        pairs = gen_instance_pairs(SAMPLE_CLASSES, compiler)
        assert all(p["pattern"] == "instance" for p in pairs)

    def test_cnl_form(self, compiler):
        pairs = gen_instance_pairs([{"term": "Process"}], compiler)
        assert all(p["cnl"] == "?x is-a Process" for p in pairs)

    def test_kif_form(self, compiler):
        pairs = gen_instance_pairs([{"term": "Process"}], compiler)
        assert all(p["kif"] == "(instance ?x Process)" for p in pairs)

    def test_multiple_nl_per_cnl(self, compiler):
        pairs = gen_instance_pairs([{"term": "Process"}], compiler)
        # At least 3 distinct NL forms for a single CNL
        assert len(pairs) >= 3

    def test_doc_added_as_nl(self, compiler):
        cls = {"term": "Entity", "doc": "The broadest possible kind of entity in the ontology."}
        pairs = gen_instance_pairs([cls], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("broadest" in nl for nl in nls)

    def test_short_doc_not_added(self, compiler):
        cls = {"term": "Process", "doc": "Short."}
        pairs_with = gen_instance_pairs([cls], compiler)
        pairs_without = gen_instance_pairs([{"term": "Process"}], compiler)
        assert len(pairs_with) == len(pairs_without)

    def test_doc_templates_can_be_disabled(self, compiler):
        cls = {"term": "Entity", "doc": "The broadest possible kind of entity in the ontology."}
        pairs = gen_instance_pairs([cls], compiler, include_doc_templates=False)
        nls = [p["nl"] for p in pairs]
        assert not any("broadest" in nl for nl in nls)

    def test_natural_language_class_names_in_nl(self, compiler):
        pairs = gen_instance_pairs([{"term": "MilitaryProcess"}], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("military process" in nl for nl in nls)

    def test_distractor_phrases_in_nl(self, compiler):
        pairs = gen_instance_pairs([{"term": "Process"}], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("in the operation" in nl for nl in nls)


# ---------------------------------------------------------------------------
# gen_subclass_pairs
# ---------------------------------------------------------------------------

class TestGenSubclassPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_subclass_pairs(SAMPLE_CLASSES, compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_subclass_pairs(SAMPLE_CLASSES, compiler)
        assert all(p["pattern"] == "subclass" for p in pairs)

    def test_cnl_form(self, compiler):
        classes = [{"term": "Process"}, {"term": "Entity"}]
        pairs = gen_subclass_pairs(classes, compiler)
        assert pairs[0]["cnl"] == "Process subclass-of Entity"

    def test_kif_form(self, compiler):
        classes = [{"term": "Process"}, {"term": "Entity"}]
        pairs = gen_subclass_pairs(classes, compiler)
        assert pairs[0]["kif"] == "(subclass Process Entity)"

    def test_n_pairs_is_n_minus_1(self, compiler):
        # 2 classes → 1 (child, parent) pair × ≥3 NL templates
        # Use Process and Entity — both core SUMO classes guaranteed to be in vocab
        classes = [{"term": "Process"}, {"term": "Entity"}]
        pairs = gen_subclass_pairs(classes, compiler)
        unique_cnl = len({p["cnl"] for p in pairs})
        assert unique_cnl == 1

    def test_single_class_produces_no_pairs(self, compiler):
        pairs = gen_subclass_pairs([{"term": "Process"}], compiler)
        assert pairs == []

    def test_natural_language_class_names_in_nl(self, compiler):
        classes = [{"term": "MilitaryUnit"}, {"term": "MilitaryOrganization"}]
        pairs = gen_subclass_pairs(classes, compiler)
        nls = [p["nl"] for p in pairs]
        assert any("military unit" in nl and "military organization" in nl for nl in nls)


# ---------------------------------------------------------------------------
# gen_binary_pairs
# ---------------------------------------------------------------------------

class TestGenBinaryPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_binary_pairs(SAMPLE_RELATIONS, compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_binary_pairs(SAMPLE_RELATIONS, compiler)
        assert all(p["pattern"] == "binary" for p in pairs)

    def test_includes_abstract_and_grounded_cnl_forms(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)  # agent
        cnls = {pair["cnl"] for pair in pairs}
        assert "agent ?x ?y" in cnls
        assert "agent Process AutonomousAgent" in cnls

    def test_grounded_nl_maps_to_grounded_cnl(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
        # Check that signature-type grounded pair is present
        sig_grounded = [p for p in pairs if p["cnl"] == "agent Process AutonomousAgent"]
        assert sig_grounded
        # Check that expanded subclass pairs are also present
        expanded = [p for p in pairs if p["cnl"] == "agent MilitaryProcess AutonomousAgent"]
        assert expanded

    def test_skips_non_binary(self, compiler):
        ternary = {"term": "agent", "arity": 3, "signature": {"1": "Process"}}
        pairs = gen_binary_pairs([ternary], compiler)
        assert pairs == []

    def test_uses_signature_types_in_nl(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("Process" in nl for nl in nls)
        assert any("AutonomousAgent" in nl for nl in nls)

    def test_doc_added_when_present(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("relates" in nl for nl in nls)

    def test_invalid_sig_class_falls_back_to_entity(self, compiler):
        rel = {
            "term": "agent",
            "arity": 2,
            "signature": {"1": "Bad-Class", "2": "Also-Bad"},
        }
        pairs = gen_binary_pairs([rel], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("Entity" in nl for nl in nls)

    def test_doc_templates_can_be_disabled(self, compiler):
        relation = {
            "term": "agent",
            "arity": 2,
            "signature": {"1": "Process", "2": "Agent"},
            "doc": "Unique binary doc text for opt-out coverage.",
        }
        pairs = gen_binary_pairs(
            [relation],
            compiler,
            include_doc_templates=False,
        )
        nls = [p["nl"] for p in pairs]
        assert not any("Unique binary doc text" in nl for nl in nls)

    def test_natural_language_type_names_in_nl(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("process" in nl.lower() and "autonomous agent" in nl.lower() for nl in nls)

    def test_relation_verb_templates_for_agent(self, compiler):
        pairs = gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("has" in nl.lower() for nl in nls)


# ---------------------------------------------------------------------------
# gen_nary_pairs
# ---------------------------------------------------------------------------

class TestGenNaryPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        assert all(p["pattern"] == "nary" for p in pairs)

    def test_cnl_form_uses_exact_bracket_syntax(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        cnls = {pair["cnl"] for pair in pairs}
        assert "between [ ?x , ?y , ?z ]" in cnls
        assert "between [ Object , Object , Object ]" in cnls

    def test_grounded_nl_maps_to_grounded_cnl(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        # Check that signature-type grounded pair is present
        sig_grounded = [p for p in pairs if p["cnl"] == "between [ Object , Object , Object ]"]
        assert sig_grounded
        # Check that expanded subclass pairs are also present
        expanded = [p for p in pairs if p["cnl"] == "between [ Weapon , MilitaryUnit , Area ]"]
        assert expanded

    def test_generated_cnl_compiles_cleanly(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        for pair in pairs:
            assert compiler.compile(pair["cnl"]) == pair["kif"]

    def test_skips_non_nary(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_RELATIONS[0]], compiler)
        assert pairs == []

    def test_doc_added_when_present(self, compiler):
        pairs = gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("Object" in nl for nl in nls)

    def test_doc_templates_can_be_disabled(self, compiler):
        pairs = gen_nary_pairs(
            [SAMPLE_NARY_RELATION],
            compiler,
            include_doc_templates=False,
        )
        nls = [p["nl"] for p in pairs]
        assert not any("places one Object between two other Objects" in nl for nl in nls)


# ---------------------------------------------------------------------------
# gen_negation_pairs
# ---------------------------------------------------------------------------

class TestGenNegationPairs:

    def test_returns_negation_pairs_for_all_atomic_forms(self, compiler):
        pairs = gen_negation_pairs(
            [{"term": "Process"}],
            [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION],
            compiler,
        )

        cnls = {pair["cnl"] for pair in pairs}
        assert "not ?x is-a Process" in cnls
        assert "not located ?x ?y" in cnls
        assert "not located Object Object" in cnls
        assert "not between [ ?x , ?y , ?z ]" in cnls
        assert "not between [ Object , Object , Object ]" in cnls

    def test_pattern_label(self, compiler):
        pairs = gen_negation_pairs(
            [{"term": "Process"}],
            [SAMPLE_NEGATION_BINARY_RELATION],
            compiler,
        )
        assert all(pair["pattern"] == "negation" for pair in pairs)

    def test_generated_cnl_compiles_cleanly(self, compiler):
        pairs = gen_negation_pairs(
            [{"term": "Process"}],
            [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION],
            compiler,
        )
        for pair in pairs:
            assert compiler.compile(pair["cnl"]) == pair["kif"]


# ---------------------------------------------------------------------------
# gen_conditional_every_pairs
# ---------------------------------------------------------------------------

class TestGenConditionalEveryPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_conditional_every_pairs(SAMPLE_RELATIONS, compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_conditional_every_pairs(SAMPLE_RELATIONS, compiler)
        assert all(p["pattern"] == "conditional" for p in pairs)

    def test_cnl_form_includes_signature_pair(self, compiler):
        pairs = gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
        cnls = {p["cnl"] for p in pairs}
        assert "every ?x is-a Process implies agent ?x AutonomousAgent" in cnls

    def test_cnl_form_includes_expanded_pairs(self, compiler):
        pairs = gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
        cnls = {p["cnl"] for p in pairs}
        assert "every ?x is-a MilitaryProcess implies agent ?x AutonomousAgent" in cnls
        assert "every ?x is-a Attack implies agent ?x MilitaryUnit" in cnls

    def test_kif_form(self, compiler):
        pairs = gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
        expected_kif = "(forall (?x) (=> (instance ?x Process) (agent ?x AutonomousAgent)))"
        sig_pairs = [p for p in pairs if p["cnl"] == "every ?x is-a Process implies agent ?x AutonomousAgent"]
        assert all(p["kif"] == expected_kif for p in sig_pairs)

    def test_natural_language_templates(self, compiler):
        pairs = gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("process" in nl.lower() and "autonomous agent" in nl.lower() for nl in nls)

    def test_verb_templates_for_agent(self, compiler):
        pairs = gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("Every" in nl and "has" in nl for nl in nls)


# ---------------------------------------------------------------------------
# gen_conditional_if_pairs
# ---------------------------------------------------------------------------

class TestGenConditionalIfPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_conditional_if_pairs(SAMPLE_RELATIONS, compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_conditional_if_pairs(SAMPLE_RELATIONS, compiler)
        assert all(p["pattern"] == "conditional" for p in pairs)

    def test_cnl_form_includes_signature_pair(self, compiler):
        pairs = gen_conditional_if_pairs([SAMPLE_RELATIONS[0]], compiler)
        cnls = {p["cnl"] for p in pairs}
        assert "if agent ?x AutonomousAgent then ?x is-a Process" in cnls

    def test_cnl_form_includes_expanded_pairs(self, compiler):
        pairs = gen_conditional_if_pairs([SAMPLE_RELATIONS[0]], compiler)
        cnls = {p["cnl"] for p in pairs}
        assert "if agent ?x MilitaryUnit then ?x is-a MilitaryProcess" in cnls

    def test_kif_form(self, compiler):
        pairs = gen_conditional_if_pairs([SAMPLE_RELATIONS[0]], compiler)
        expected_kif = "(=> (agent ?x AutonomousAgent) (instance ?x Process))"
        sig_pairs = [p for p in pairs if p["cnl"] == "if agent ?x AutonomousAgent then ?x is-a Process"]
        assert all(p["kif"] == expected_kif for p in sig_pairs)


# ---------------------------------------------------------------------------
# gen_existential_pairs
# ---------------------------------------------------------------------------

class TestGenExistentialPairs:

    def test_returns_nonempty(self, compiler):
        pairs = gen_existential_pairs(SAMPLE_CLASSES, compiler)
        assert len(pairs) > 0

    def test_pattern_label(self, compiler):
        pairs = gen_existential_pairs(SAMPLE_CLASSES, compiler)
        assert all(p["pattern"] == "existential" for p in pairs)

    def test_cnl_form(self, compiler):
        pairs = gen_existential_pairs([{"term": "Process"}], compiler)
        assert all(p["cnl"] == "some ?x is-a Process" for p in pairs)

    def test_kif_form(self, compiler):
        pairs = gen_existential_pairs([{"term": "Process"}], compiler)
        assert all(p["kif"] == "(exists (?x) (instance ?x Process))" for p in pairs)

    def test_multiple_nl_forms(self, compiler):
        pairs = gen_existential_pairs([{"term": "Process"}], compiler)
        assert len(pairs) >= 5

    def test_natural_language_class_names_in_nl(self, compiler):
        pairs = gen_existential_pairs([{"term": "MilitaryProcess"}], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("military process" in nl for nl in nls)

    def test_distractor_phrases_in_nl(self, compiler):
        pairs = gen_existential_pairs([{"term": "Process"}], compiler)
        nls = [p["nl"] for p in pairs]
        assert any("in the operation" in nl for nl in nls)


# ---------------------------------------------------------------------------
# Integration: output JSONL
# ---------------------------------------------------------------------------

class TestOutputIntegration:

    def test_all_pairs_have_required_fields(self, compiler):
        """All generators produce complete records."""
        all_pairs = (
            gen_instance_pairs([{"term": "Process"}], compiler)
            + gen_subclass_pairs([{"term": "Process"}, {"term": "Entity"}], compiler)
            + gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
            + gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_conditional_if_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_existential_pairs([{"term": "Process"}], compiler)
            + gen_negation_pairs([{"term": "Process"}], [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION], compiler)
        )
        required = {"nl", "cnl", "kif", "pattern"}
        for pair in all_pairs:
            assert set(pair.keys()) == required, f"Missing fields in: {pair}"

    def test_kif_round_trips(self, compiler):
        """Every generated KIF is exactly what CNLCompiler produces for the CNL."""
        for rec in SAMPLE_CLASSES:
            cnl = f"?x is-a {rec['term']}"
            try:
                expected_kif = compiler.compile(cnl)
            except Exception:
                continue
            pairs = gen_instance_pairs([rec], compiler)
            assert all(p["kif"] == expected_kif for p in pairs)

    def test_jsonl_serialisable(self, compiler):
        """All pairs serialise cleanly to JSON."""
        pairs = gen_instance_pairs(SAMPLE_CLASSES, compiler)
        for pair in pairs:
            s = json.dumps(pair, ensure_ascii=False)
            parsed = json.loads(s)
            assert parsed == pair

    def test_pattern_values_are_valid(self, compiler):
        valid_patterns = {"instance", "subclass", "binary", "nary", "conditional", "existential", "negation"}
        all_pairs = (
            gen_instance_pairs([{"term": "Process"}], compiler)
            + gen_subclass_pairs([{"term": "Process"}, {"term": "Entity"}], compiler)
            + gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
            + gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_conditional_if_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_existential_pairs([{"term": "Process"}], compiler)
            + gen_negation_pairs([{"term": "Process"}], [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION], compiler)
        )
        for pair in all_pairs:
            assert pair["pattern"] in valid_patterns


class TestBalancedSampling:

    def test_balanced_limit_caps_each_pattern_group(self):
        patterns = [
            "instance",
            "subclass",
            "binary",
            "conditional",
            "existential",
            "nary",
            "negation",
        ]
        pairs = []
        for pattern in patterns:
            for index in range(5):
                pairs.append(
                    {
                        "nl": f"{pattern} nl {index}",
                        "cnl": f"{pattern} cnl {index}",
                        "kif": f"({pattern} {index})",
                        "pattern": pattern,
                    }
                )

        selected = select_output_pairs(pairs, limit=12, balanced=True)
        counts = count_pairs_by_pattern(selected)
        per_pattern_cap = 12 // len(patterns)

        assert len(selected) == per_pattern_cap * len(patterns)
        for pattern in patterns:
            assert counts.get(pattern, 0) <= per_pattern_cap

    def test_balanced_counts_include_nary_bucket(self, compiler):
        pairs = (
            gen_instance_pairs([{"term": "Process"}], compiler)
            + gen_subclass_pairs([{"term": "Process"}, {"term": "Entity"}], compiler)
            + gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
            + gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_existential_pairs([{"term": "Process"}], compiler)
            + gen_negation_pairs([{"term": "Process"}], [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION], compiler)
        )

        selected = select_output_pairs(pairs, limit=12, balanced=True)
        counts = count_pairs_by_pattern(selected)

        assert "nary" in counts
        assert counts["nary"] <= 12 // len(counts)

    def test_balanced_counts_include_negation_bucket(self, compiler):
        pairs = (
            gen_instance_pairs([{"term": "Process"}], compiler)
            + gen_subclass_pairs([{"term": "Process"}, {"term": "Entity"}], compiler)
            + gen_binary_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_nary_pairs([SAMPLE_NARY_RELATION], compiler)
            + gen_conditional_every_pairs([SAMPLE_RELATIONS[0]], compiler)
            + gen_existential_pairs([{"term": "Process"}], compiler)
            + gen_negation_pairs([{"term": "Process"}], [SAMPLE_NEGATION_BINARY_RELATION, SAMPLE_NARY_RELATION], compiler)
        )

        selected = select_output_pairs(pairs, limit=14, balanced=True)
        counts = count_pairs_by_pattern(selected)

        assert "negation" in counts
        assert counts["negation"] <= 14 // len(counts)
