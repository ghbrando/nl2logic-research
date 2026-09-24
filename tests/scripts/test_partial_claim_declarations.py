from pathlib import Path

from scripts.score_partial_claim_declarations import score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter1-partial-claim-development-source"
RUN = ROOT / "results/diagnostics/partial_claim_memory_cpu_20260923_declarations"


def test_saved_declaration_probe_separates_validity_from_source_support():
    scored, summary = score(
        SOURCE / "sources.jsonl",
        SOURCE / "manifest.json",
        SOURCE / "provisional_declarations.json",
        ROOT / "data/benchmarks/fm2-0/development_partial_claims_20260923.json",
        RUN / "predictions.jsonl",
        RUN / "manifest.json",
    )
    assert summary["oracle_declarations_applied"] == 3
    assert summary["changed_output_count"] == 2
    assert summary["by_mode"]["baseline"]["exact_ai_label_matches"] == 1
    assert summary["by_mode"]["retrieved"]["exact_ai_label_matches"] == 1
    assert summary["by_mode"]["baseline"]["grounder_accepted"] == 0
    assert summary["by_mode"]["retrieved"]["grounder_accepted"] == 0
    assert all(row["grounder_accepted"] is False for row in scored if row["declared_symbol"])
    assert [row["paragraph"] for row in scored if row["gold_kif_exact"]] == ["1-93", "1-93"]


def test_passage_only_declared_gate_rejects_incidental_and_reversed_outputs():
    scored, summary = score(
        SOURCE / "sources.jsonl",
        SOURCE / "manifest.json",
        SOURCE / "provisional_declarations.json",
        ROOT / "data/benchmarks/fm2-0/development_partial_claims_20260923.json",
        RUN / "predictions.jsonl",
        RUN / "manifest.json",
    )
    reasons = {(row["paragraph"], row["mode"]): row["declared_gate_reason"]
               for row in scored if row["declared_symbol"]}
    assert reasons == {
        ("1-79", "baseline"): "undeclared_child",
        ("1-79", "retrieved"): "reversed_direction",
        ("1-88", "baseline"): "unstated_parent",
        ("1-88", "retrieved"): "reversed_direction",
        ("1-93", "baseline"): None,
        ("1-93", "retrieved"): None,
    }
    for mode in ("baseline", "retrieved"):
        assert summary["by_mode"][mode]["declared_gate_accepted_other_formula"] == 0


GATE_RUN = ROOT / "results/diagnostics/partial_claim_definition_gate_cpu_20260923"


def test_definition_child_decoding_run_and_prior_claim_memory():
    from src.reasoning.claim_memory import load_prior_claim_memory, render_memory_prompt

    scored, summary = score(
        SOURCE / "sources.jsonl",
        SOURCE / "manifest.json",
        SOURCE / "provisional_declarations.json",
        ROOT / "data/benchmarks/fm2-0/development_partial_claims_20260923.json",
        GATE_RUN / "predictions.jsonl",
        GATE_RUN / "manifest.json",
    )
    assert summary["by_mode"]["rules"]["declared_gate_accepted_ai_label_match"] == 2
    for mode in ("baseline", "retrieved"):
        assert summary["by_mode"][mode]["declared_gate_accepted"] == 1
        assert summary["by_mode"][mode]["declared_gate_accepted_other_formula"] == 0
    assert summary["changed_output_count"] == 0
    assert not any(row["predicted_cnl"] and row["predicted_cnl"].startswith(("Source ", "Process ")) for row in scored)

    memory = load_prior_claim_memory(GATE_RUN / "scoring/scored.jsonl")
    assert [statement.kif for statement in memory] == [
        "(subclass OpenSourceIntelligenceProduct IntelligenceProduct)",
        "(subclass BiometricsProcess Process)",
    ]
    assert "Prior accepted claim: (subclass BiometricsProcess Process)" in render_memory_prompt("x", memory[1:])


def test_unseen_run_reports_unsupported_gate_acceptances():
    unseen = ROOT / "data/benchmarks/fm2-0/chapters3plus-definition-unseen-source"
    run = ROOT / "results/diagnostics/unseen_definition_cpu_20260923"
    scored, summary = score(
        unseen / "sources.jsonl", unseen / "manifest.json", unseen / "proposed_declarations.json",
        unseen / "ai_reviewed_claims_20260923.json", run / "predictions.jsonl", run / "manifest.json",
    )
    assert summary["label_status"] == "ai_reviewed_frozen_pre_target_query"
    assert summary["declarations_applied"] == 101
    saved = {mode: (values["saved_gate_accepted"], values["saved_gate_accepted_other_formula"])
             for mode, values in summary["by_mode"].items()}
    assert saved == {"rules": (8, 6), "baseline": (3, 1), "retrieved": (4, 2)}
    # The head-noun gate, fixed afterwards on synthetic frames, keeps the two
    # label matches and rejects every unsupported acceptance.
    current = {mode: (values["declared_gate_accepted_ai_label_match"], values["declared_gate_accepted_other_formula"])
               for mode, values in summary["by_mode"].items()}
    assert current == {"rules": (2, 0), "baseline": (2, 0), "retrieved": (2, 0)}
    assert summary["changed_output_count"] == 1
    # Modifier heads named existing classes and passed the frozen gate.
    assert {row["predicted_cnl"] for row in scored if row["mode"] == "rules" and row.get("saved_gate_accepted")
            and not row["gold_kif_exact"]} == {
        "StaffKey subclass-of Key", "RiskManagementArmy subclass-of Army",
        "StructureOfATacticalCpOrganic subclass-of Organic", "CorpsArmy subclass-of Army",
        "BctArmy subclass-of Army", "ThreeBasicFriendlyDefensiveOperationsArea subclass-of Area",
    }


def test_appendix_run_after_head_noun_revision():
    source = ROOT / "data/benchmarks/fm2-0/appendices-definition-unseen-source"
    run = ROOT / "results/diagnostics/appendix_definition_cpu_20260923"
    scored, summary = score(
        source / "sources.jsonl", source / "manifest.json", source / "proposed_declarations.json",
        source / "ai_reviewed_claims_20260923.json", run / "predictions.jsonl", run / "manifest.json",
    )
    counts = {mode: (values["declared_gate_accepted_ai_label_match"], values["declared_gate_accepted_other_formula"])
              for mode, values in summary["by_mode"].items()}
    assert counts == {"rules": (2, 1), "baseline": (2, 0), "retrieved": (2, 0)}
    assert summary["changed_output_count"] == 0
    assert [row["predicted_cnl"] for row in scored if row["declared_gate_accepted"] and not row["gold_kif_exact"]] == [
        "SituationalUnderstandingProduct subclass-of Product"]
