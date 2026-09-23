"""Score saved source-only predictions against frozen AI claim labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_claim_packet import validate as validate_labels
from src.compiler.compiler import CNLCompiler
from src.eval.comparison import restates_ontology
from src.ingest.grounding import DoctrineGrounder


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def score(
    selected_path: Path,
    labels_path: Path,
    review_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
) -> tuple[list[dict], dict]:
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review["status"] != "ai_reviewed_frozen_pre_target_query" or text_sha256(labels_path) != review["label_packet_sha256"]:
        raise ValueError("AI review is not the frozen label packet")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    validate_labels(labels)
    label_by_id = {row["record_id"]: row for row in labels["claims"]}
    selected = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_id = {row["record_id"]: row for row in selected}
    run = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if run["status"] != "completed" or text_sha256(predictions_path) != run["predictions_sha256"]:
        raise ValueError("Prediction run is incomplete or predictions changed")
    if run["selected_sha256"] != text_sha256(selected_path) or run["source_selection_manifest_sha256"] != review["source_selection_manifest_sha256"]:
        raise ValueError("Predictions were made from a different source selection")
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(predictions) != len(selected) or len({row["record_id"] for row in predictions}) != len(selected):
        raise ValueError("Prediction count or IDs differ from source selection")
    if set(source_by_id) != set(label_by_id) or set(source_by_id) != {row["record_id"] for row in predictions}:
        raise ValueError("Prediction, source, and label IDs differ")
    compiler = CNLCompiler()
    grounder = DoctrineGrounder(allow_background_axioms=False)
    scored = []
    for prediction in predictions:
        rid = prediction["record_id"]
        source = source_by_id[rid]["source_excerpt"]
        if prediction["source_sha256"] != hashlib.sha256(source.encode("utf-8")).hexdigest():
            raise ValueError(f"{rid}: prediction source hash differs")
        label = label_by_id[rid]
        for method in ("rules", "raw"):
            cnl = prediction.get(f"{method}_cnl")
            error = prediction.get(f"{method}_error")
            result = {
                "record_id": rid,
                "paragraph": label["paragraph"],
                "method": method,
                "label_decision": label["decision"],
                "source_excerpt": source,
                "predicted_cnl": cnl,
                "compiled_kif": None,
                "route": "abstain",
                "reason": None,
                "error": error,
                "gold_kif_exact": False,
                "ontology_restated": None,
            }
            if error:
                result["route"] = "error"
            elif cnl:
                try:
                    kif = compiler.compile(cnl, require_closed=True, require_argument_kinds=True)
                except Exception as exc:
                    result.update(route="compile_error", error=f"{type(exc).__name__}: {exc}")
                else:
                    result["compiled_kif"] = kif
                    result["gold_kif_exact"] = label["decision"] == "positive" and kif == label["kif"]
                    result["ontology_restated"] = restates_ontology(kif)
                    try:
                        assessment = grounder.assess(
                            cnl=cnl, original_text=source, normalized_text=source,
                            linked_text=source, subclaim_text=source,
                        )
                        result.update(route="accepted" if assessment.accepted else "review",
                                      reason=assessment.reason)
                    except Exception as exc:
                        result.update(route="error", error=f"{type(exc).__name__}: {exc}")
            scored.append(result)
    by_method = {}
    for method in ("rules", "raw"):
        rows = [row for row in scored if row["method"] == method]
        negative = [row for row in rows if row["label_decision"] == "abstain"]
        by_method[method] = {
            "routes": dict(Counter(row["route"] for row in rows)),
            "false_accepts_on_28_ai_abstentions": sum(row["route"] == "accepted" for row in negative),
            "accepted_ontology_restatements": sum(row["route"] == "accepted" and row["ontology_restated"] is True for row in rows),
            "one_positive_exact_kif": any(row["gold_kif_exact"] for row in rows),
        }
    summary = {
        "label_status": review["status"],
        "source_selection_manifest_sha256": review["source_selection_manifest_sha256"],
        "label_packet_sha256": review["label_packet_sha256"],
        "predictions_sha256": run["predictions_sha256"],
        "adapter_sha256": run["adapter_sha256"],
        "source_passages": len(selected),
        "ai_positives": sum(row["decision"] == "positive" for row in labels["claims"]),
        "ai_abstentions": sum(row["decision"] == "abstain" for row in labels["claims"]),
        "guarded_input_gate_abstentions": sum(bool(row["gate_reasons"]) for row in predictions),
        "methods": by_method,
        "interpretation": "Exploratory AI-label diagnostic. Raw greedily generates from full paragraphs despite the production unsupported-input gate. One positive cannot support a stable recall estimate. Accepted routes require manual semantic review.",
    }
    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scored, summary = score(args.selected, args.labels, args.review_manifest,
                            args.predictions, args.prediction_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "scored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
