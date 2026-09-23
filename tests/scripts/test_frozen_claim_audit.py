import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_frozen_claim_predictions import load_sources, text_sha256
from scripts.score_frozen_claim_predictions import score
from scripts.score_frozen_guarded_predictions import score as score_guarded


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/benchmarks/fm2-0/chapter2-2023-claim-eval-source"


def test_prediction_input_is_unlabeled_frozen_source():
    rows = load_sources(SOURCE / "selected_passages.jsonl", SOURCE / "manifest.json")
    assert len(rows) == 29
    assert all(row["decision"] is None for row in rows)
    assert all("cnl" not in row and "kif" not in row for row in rows)


def _write_all_abstain_predictions(tmp_path: Path) -> tuple[Path, Path]:
    rows = load_sources(SOURCE / "selected_passages.jsonl", SOURCE / "manifest.json")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps({
        "record_id": row["record_id"],
        "source_sha256": hashlib.sha256(row["source_excerpt"].encode("utf-8")).hexdigest(),
        "gate_reasons": ["multiple sentences detected"],
        "rules_cnl": None, "rules_error": None,
        "raw_cnl": None, "raw_error": None,
    }) + "\n" for row in rows), encoding="utf-8")
    review = json.loads((SOURCE / "review_manifest.json").read_text(encoding="utf-8"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "completed",
        "predictions_sha256": text_sha256(predictions),
        "selected_sha256": text_sha256(SOURCE / "selected_passages.jsonl"),
        "source_selection_manifest_sha256": review["source_selection_manifest_sha256"],
        "adapter_sha256": "synthetic-test-adapter",
    }), encoding="utf-8")
    return predictions, manifest


def test_score_uses_saved_predictions_and_frozen_labels(tmp_path):
    predictions, manifest = _write_all_abstain_predictions(tmp_path)
    scored, summary = score(
        SOURCE / "selected_passages.jsonl",
        SOURCE / "ai_reviewed_claims_20260923.json",
        SOURCE / "review_manifest.json",
        predictions, manifest,
    )
    assert len(scored) == 58
    assert summary["ai_positives"] == 1
    assert summary["ai_abstentions"] == 28
    assert summary["methods"]["raw"]["false_accepts_on_28_ai_abstentions"] == 0


def test_score_rejects_prediction_from_another_source(tmp_path):
    predictions, manifest = _write_all_abstain_predictions(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_sha256"] = "0" * 64
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    run = json.loads(manifest.read_text(encoding="utf-8"))
    run["predictions_sha256"] = text_sha256(predictions)
    manifest.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction source hash differs"):
        score(SOURCE / "selected_passages.jsonl", SOURCE / "ai_reviewed_claims_20260923.json",
              SOURCE / "review_manifest.json", predictions, manifest)


def test_guarded_audit_records_decoder_abstention_after_input_gate():
    run = ROOT / "results/diagnostics/chapter2_cpu_guarded_20260923"
    scored, summary = score_guarded(
        SOURCE / "selected_passages.jsonl",
        SOURCE / "ai_reviewed_claims_20260923.json",
        SOURCE / "review_manifest.json",
        run / "guarded_predictions.jsonl",
        run / "manifest.json",
    )
    assert summary["routes"] == {"input_gate_abstain": 28, "lexical_abstain": 1}
    assert summary["false_accepts_on_ai_abstentions"] == 0
    assert not summary["one_positive_exact_kif"]
    assert [row["paragraph"] for row in scored if row["route"] == "lexical_abstain"] == ["2-10"]
