from src.eval.evaluate import (
    DEFAULT_GOLD_PATH,
    GoldPair,
    evaluate_pairs,
    load_gold_pairs,
    validate_gold_pairs,
)


class TestGoldSet:
    def test_load_gold_pairs_has_36_records_and_balanced_patterns(self):
        pairs = load_gold_pairs(DEFAULT_GOLD_PATH)

        assert len(pairs) == 36

        counts: dict[str, int] = {}
        for pair in pairs:
            counts[pair.pattern] = counts.get(pair.pattern, 0) + 1

        assert counts == {
            "instance": 6,
            "subclass": 6,
            "binary": 6,
            "conditional": 6,
            "existential": 6,
            "negation": 6,
        }

    def test_gold_pairs_compile_cleanly(self):
        pairs = load_gold_pairs(DEFAULT_GOLD_PATH)

        validate_gold_pairs(pairs)


class TestEvaluatePairs:
    def test_oracle_mode_is_perfect(self):
        pairs = load_gold_pairs(DEFAULT_GOLD_PATH)

        report = evaluate_pairs(pairs)

        assert report.total == 36
        assert report.cnl_accuracy == 1.0
        assert report.kif_accuracy == 1.0
        assert report.abstain_rate == 0.0

    def test_custom_predictor_tracks_mismatch_and_abstain(self):
        pairs = [
            GoldPair(
                nl="A military unit is a type of military organization.",
                cnl="MilitaryUnit subclass-of MilitaryOrganization",
                kif="(subclass MilitaryUnit MilitaryOrganization)",
                pattern="subclass",
            ),
            GoldPair(
                nl="There exists a weapon in the arsenal.",
                cnl="some ?x is-a Weapon",
                kif="(exists (?x) (instance ?x Weapon))",
                pattern="existential",
            ),
        ]

        predictions = {
            pairs[0].nl: "MilitaryUnit subclass-of Organization",
            pairs[1].nl: None,
        }

        report = evaluate_pairs(
            pairs,
            predictor=lambda pair: predictions[pair.nl],
        )

        assert report.total == 2
        assert report.cnl_exact == 0
        assert report.kif_exact == 0
        assert report.abstained == 1
        assert report.pattern_breakdown["subclass"].total == 1
        assert report.pattern_breakdown["subclass"].cnl_accuracy == 0.0
        assert report.pattern_breakdown["existential"].abstain_rate == 1.0
