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
