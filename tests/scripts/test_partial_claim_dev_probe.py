from pathlib import Path

from scripts.score_partial_claim_dev_probe import score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter1-partial-claim-development-source"
RUN = ROOT / "results/diagnostics/partial_claim_dev_cpu_20260923_02"


def test_saved_dev_probe_scores_only_closed_valid_outputs():
    scored, summary = score(
        SOURCE / "sources.jsonl",
        SOURCE / "manifest.json",
        ROOT / "data/benchmarks/fm2-0/development_partial_claims_20260923.json",
        RUN / "predictions.jsonl",
        RUN / "manifest.json",
    )
    assert summary["routes"] == {
        "extraction_abstain": 11,
        "decoder_abstain": 2,
        "compile_error": 1,
    }
    assert summary["exact_gold_kif"] == 0
    assert [row["error"] for row in scored if row["paragraph"] == "1-93"] == [
        "ValueError: Unbound variables: ?x"
    ]
