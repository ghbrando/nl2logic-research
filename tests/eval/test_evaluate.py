from src.eval.evaluate import (
    DEFAULT_GOLD_PATH,
    GoldPair,
    evaluate_pairs,
    load_gold_pairs,
    parse_args,
    print_report,
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
        assert len(report.examples) == 36
        assert report.mismatches == ()

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
        assert len(report.examples) == 2
        assert len(report.mismatches) == 2
        assert report.mismatches[0].predicted_cnl == "MilitaryUnit subclass-of Organization"
        assert report.mismatches[0].predicted_kif == "(subclass MilitaryUnit Organization)"
        assert report.mismatches[0].compile_error is None
        assert report.mismatches[1].abstained is True
        assert report.mismatches[1].predicted_cnl is None

    def test_compile_errors_are_captured_in_example_results(self):
        pairs = [
            GoldPair(
                nl="A weapon does not have a region.",
                cnl="not relatedInternalConcept Weapon Region",
                kif="(not (relatedInternalConcept Weapon Region))",
                pattern="negation",
            )
        ]

        report = evaluate_pairs(
            pairs,
            predictor=lambda pair: "not valid cnl",
        )

        assert report.total == 1
        assert report.kif_exact == 0
        assert len(report.mismatches) == 1
        assert report.mismatches[0].predicted_cnl == "not valid cnl"
        assert report.mismatches[0].predicted_kif is None
        assert report.mismatches[0].compile_error is not None
        assert "Unexpected" in report.mismatches[0].compile_error


class TestCliReporting:
    def test_parse_args_defaults_match_expected_reporting_flags(self):
        args = parse_args([])

        assert args.show_misses is False
        assert args.max_misses == 0

    def test_print_report_shows_mismatches(self, capsys):
        pair = GoldPair(
            nl="A military unit is a type of military organization.",
            cnl="MilitaryUnit subclass-of MilitaryOrganization",
            kif="(subclass MilitaryUnit MilitaryOrganization)",
            pattern="subclass",
        )
        report = evaluate_pairs([pair], predictor=lambda _: "MilitaryUnit subclass-of Organization")

        print_report(report, show_misses=True)
        output = capsys.readouterr().out

        assert "Mismatches:" in output
        assert "Predicted CNL: MilitaryUnit subclass-of Organization" in output
        assert "Predicted KIF: (subclass MilitaryUnit Organization)" in output

    def test_print_report_honors_max_misses(self, capsys):
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
        report = evaluate_pairs(pairs, predictor=lambda pair: predictions[pair.nl])

        print_report(report, show_misses=True, max_misses=1)
        output = capsys.readouterr().out

        assert output.count("Pattern:") == 1
        assert "There exists a weapon in the arsenal." not in output

CHALLENGE_GOLD_PATH = DEFAULT_GOLD_PATH.parent / "challenge_pairs.jsonl"


class TestChallengeSet:
    def test_load_challenge_pairs_has_21_records_and_balanced_patterns(self):
        pairs = load_gold_pairs(CHALLENGE_GOLD_PATH)

        assert len(pairs) == 21

        counts: dict[str, int] = {}
        for pair in pairs:
            counts[pair.pattern] = counts.get(pair.pattern, 0) + 1

        assert counts == {
            "instance": 3,
            "subclass": 3,
            "binary": 3,
            "conditional": 3,
            "existential": 3,
            "negation": 3,
            "nary": 3,
        }

    def test_challenge_pairs_compile_cleanly(self):
        pairs = load_gold_pairs(CHALLENGE_GOLD_PATH)

        validate_gold_pairs(pairs)