import scripts.diagnose_constrained_eval as diagnose


class TestDiagnoseConstrainedEval:
    def test_parse_args_defaults_to_full_grammar_mode(self):
        args = diagnose.parse_args([])

        assert args.grammar_mode == "full"
        assert args.max_new_tokens == 64
        assert args.pair_index == 0

    def test_build_control_grammar_uses_exact_literal(self):
        grammar = diagnose.build_control_grammar("some ?x is-a Process")

        assert grammar == 'root ::= "some ?x is-a Process"'

    def test_select_probe_grammar_control_uses_gold_cnl(self):
        grammar, label = diagnose.select_probe_grammar(
            gold_cnl="?x is-a MilitaryProcess",
            grammar_mode="control",
        )

        assert grammar == 'root ::= "?x is-a MilitaryProcess"'
        assert label == "single-literal control grammar"

    def test_select_probe_grammar_full_uses_builder(self, monkeypatch):
        monkeypatch.setattr(diagnose, "build_xgrammar_grammar", lambda: 'root ::= "ok"')

        grammar, label = diagnose.select_probe_grammar(
            gold_cnl="?x is-a Process",
            grammar_mode="full",
        )

        assert grammar == 'root ::= "ok"'
        assert label == "full closed-vocabulary grammar"
