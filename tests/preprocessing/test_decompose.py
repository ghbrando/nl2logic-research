import pytest

from src.preprocessing.decompose import decompose


class TestDecompose:
    @pytest.mark.parametrize("nl", [
        "A weapon is an artifact or a human is an animal.",
        "A weapon is an artifact; or a human is an animal.",
        "If the brigade advances, and the battalion defends, the commander withdraws.",
        "The commander does not advance, and the battalion defends.",
        "The commander doesn't advance, and the battalion defends.",
        "No commander advances, and the battalion defends.",
    ])
    def test_scoped_clauses_never_become_independent_claims(self, nl):
        # Unsupported whole sentences may abstain, but no fragment is safe.
        parts = decompose(nl)
        assert parts == [] or parts == [nl]

    def test_single_sentence_passes_through_unchanged(self):
        nl = "Every Process is in the agent relation with an Agent."

        assert decompose(nl) == [nl]

    def test_compound_sentence_splits_on_conjunction(self):
        nl = "The brigade advances, and the battalion defends."

        assert decompose(nl) == [
            "The brigade advances.",
            "The battalion defends.",
        ]

    def test_relative_clause_splits_into_two_subclaims(self):
        nl = "The commander, who leads the unit, attacks."

        assert decompose(nl) == [
            "The commander attacks.",
            "The commander leads the unit.",
        ]

    def test_explicit_list_structure_splits_on_semicolons(self):
        nl = "The force secures the bridge; the force protects the flank; and the force advances."

        assert decompose(nl) == [
            "The force secures the bridge.",
            "The force protects the flank.",
            "The force advances.",
        ]

    def test_subclaims_that_trigger_abstain_are_dropped(self, caplog):
        nl = "The brigade advances, and it withdraws."

        with caplog.at_level("WARNING"):
            parts = decompose(nl)

        assert parts == ["The brigade advances."]
        assert "Skipping decomposed subclaim that still appears unsupported" in caplog.text

    @pytest.mark.parametrize("nl", ["", "   \n\t  "])
    def test_empty_and_whitespace_inputs_return_empty_list(self, nl):
        assert decompose(nl) == []
