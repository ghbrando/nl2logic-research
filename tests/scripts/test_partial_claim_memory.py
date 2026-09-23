import json
from pathlib import Path

from scripts.compare_partial_claim_memory import compare


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/diagnostics"


def test_saved_memory_ablation_reports_pre_generation_block():
    summary = compare(
        RESULTS / "partial_claim_memory_cpu_20260923_baseline",
        RESULTS / "partial_claim_memory_cpu_20260923_retrieved",
    )
    assert summary["rows_with_retrieved_memory"] == 2
    assert summary["memory_blocked_before_generation"] == 2
    assert summary["memory_entered_generation"] == 0
    assert summary["changed_output_count"] == 0


def test_control_reaches_model_but_rejects_unbound_output():
    control = json.loads((RESULTS / "partial_claim_memory_cpu_20260923_control/control.json").read_text(encoding="utf-8"))
    baseline, memory = control["arms"]
    assert baseline["model_prompt_sha256"] == control["source_prompt_sha256"]
    assert memory["model_prompt_sha256"] != control["source_prompt_sha256"]
    assert memory["memory_statement_ids"] == ["doctrine-domain:195"]
    assert [arm["error"] for arm in control["arms"]] == [
        "ValueError: Unbound variables: ?x",
        "ValueError: Unbound variables: ?x",
    ]
