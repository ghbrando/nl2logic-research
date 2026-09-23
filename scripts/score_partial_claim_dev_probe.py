"""Score saved, label-isolated development predictions against AI review."""
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

from scripts.run_frozen_claim_predictions import text_sha256
from scripts.validate_claim_packet import validate
from src.compiler.compiler import CNLCompiler
from src.ingest.grounding import DoctrineGrounder


def score(sources_path: Path, source_manifest_path: Path, packet_path: Path,
          predictions_path: Path, prediction_manifest_path: Path) -> tuple[list[dict], dict]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    run = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if text_sha256(sources_path) != source_manifest["sources_sha256"]:
        raise ValueError("Development sources changed")
    if text_sha256(packet_path) != source_manifest["packet_sha256"]:
        raise ValueError("Development label packet changed")
    if (run["status"] != "completed"
            or run["source_manifest_sha256"] != text_sha256(source_manifest_path)
            or run["sources_sha256"] != text_sha256(sources_path)
            or run["predictions_sha256"] != text_sha256(predictions_path)):
        raise ValueError("Prediction run incomplete or inputs/outputs changed")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    validate(packet)
    labels = {row["record_id"]: row for row in packet["claims"]}
    sources = {row["record_id"]: row for row in map(json.loads, sources_path.read_text(encoding="utf-8").splitlines())}
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(predictions) != run["prediction_count"] or len(predictions) != source_manifest["source_count"]
            or len({row["record_id"] for row in predictions}) != len(predictions)
            or set(labels) != set(sources) or set(sources) != {row["record_id"] for row in predictions}):
        raise ValueError("Prediction, source, and label IDs or counts differ")
    compiler = CNLCompiler()
    grounder = DoctrineGrounder(allow_background_axioms=False)
    scored = []
    for row in predictions:
        rid = row["record_id"]
        source = sources[rid]
        label = labels[rid]
        if (row["source_sha256"] != hashlib.sha256(source["source_excerpt"].encode("utf-8")).hexdigest()
                or row["source_sentence_sha256"] != hashlib.sha256(source["source_sentence"].encode("utf-8")).hexdigest()):
            raise ValueError(f"{rid}: source hash differs")
        extraction = row["extraction"]
        if extraction["status"] == "candidate":
            start, end = extraction["start"], extraction["end"]
            if source["source_excerpt"][start:end] != extraction["source_span"]:
                raise ValueError(f"{rid}: extracted span is not exact source text")
        elif row["constrained_cnl"]:
            raise ValueError(f"{rid}: generated CNL without an extracted candidate")
        cnl = row["constrained_cnl"]
        result = {
            "record_id": rid,
            "paragraph": label["paragraph"],
            "label_decision": label["decision"],
            "candidate_text": extraction["candidate_text"],
            "extraction_reason": extraction["reason"],
            "predicted_cnl": cnl,
            "compiled_kif": None,
            "route": "extraction_abstain" if extraction["status"] == "abstain" else "candidate",
            "reason": None,
            "error": row["error"],
            "gold_kif_exact": False,
        }
        if result["route"] == "candidate":
            if extraction["gate_reasons"]:
                result.update(route="input_gate_abstain", reason="; ".join(extraction["gate_reasons"]))
            elif row["decoder_abstention"]:
                result.update(route="decoder_abstain", reason=row["decoder_abstention"])
            elif row["error"]:
                result["route"] = "error"
            elif not cnl:
                result["route"] = "empty_output"
            else:
                try:
                    kif = compiler.compile(cnl, require_closed=True, require_argument_kinds=True)
                except Exception as exc:
                    result.update(route="compile_error", error=f"{type(exc).__name__}: {exc}")
                else:
                    result["compiled_kif"] = kif
                    result["gold_kif_exact"] = label["decision"] == "positive" and kif == label["kif"]
                    try:
                        assessment = grounder.assess(
                            cnl=cnl, original_text=source["source_excerpt"],
                            normalized_text=source["source_sentence"],
                            linked_text=source["source_sentence"],
                            subclaim_text=extraction["candidate_text"],
                        )
                        result.update(route="accepted" if assessment.accepted else "review", reason=assessment.reason)
                    except Exception as exc:
                        result.update(route="error", error=f"{type(exc).__name__}: {exc}")
        scored.append(result)
    summary = {
        "label_status": "ai_reviewed_development_not_gold",
        "source_manifest_sha256": text_sha256(source_manifest_path),
        "predictions_sha256": run["predictions_sha256"],
        "adapter_sha256": run["adapter_sha256"],
        "source_passages": len(scored),
        "ai_positives": sum(row["decision"] == "positive" for row in packet["claims"]),
        "ai_abstentions": sum(row["decision"] == "abstain" for row in packet["claims"]),
        "extracted_candidates": sum(row["candidate_text"] is not None for row in scored),
        "candidates_on_ai_positives": sum(row["candidate_text"] is not None and row["label_decision"] == "positive" for row in scored),
        "candidates_on_ai_abstentions": sum(row["candidate_text"] is not None and row["label_decision"] == "abstain" for row in scored),
        "routes": dict(Counter(row["route"] for row in scored)),
        "accepted_on_ai_abstentions": sum(row["route"] == "accepted" and row["label_decision"] == "abstain" for row in scored),
        "exact_gold_kif": sum(row["gold_kif_exact"] for row in scored),
        "interpretation": "Development-set diagnostic only. Candidate selection is not semantic entailment; AI-only labels and a deliberately selected set do not support generalization.",
    }
    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scored, summary = score(args.sources, args.source_manifest, args.packet,
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
