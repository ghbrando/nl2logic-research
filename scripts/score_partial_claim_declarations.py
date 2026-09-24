"""Score saved oracle-declaration development outputs after label-free inference."""
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
from src.eval.comparison import restates_ontology
from src.ingest.grounding import DoctrineGrounder, assess_declared_definition
from scripts.run_partial_claim_declaration_probe import declaration_parents
from src.preprocessing.declarations import load_declaration_registry, match_declaration


def score(sources_path: Path, source_manifest_path: Path, registry_path: Path,
          packet_path: Path, predictions_path: Path, prediction_manifest_path: Path) -> tuple[list[dict], dict]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    run = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if (text_sha256(sources_path) != source_manifest["sources_sha256"]
            or ("packet_sha256" in source_manifest and text_sha256(packet_path) != source_manifest["packet_sha256"])):
        raise ValueError("Development source or labels changed")
    if (run["status"] != "completed"
            or run["sources_sha256"] != text_sha256(sources_path)
            or run["source_manifest_sha256"] != text_sha256(source_manifest_path)
            or run["registry_sha256"] != text_sha256(registry_path)
            or run["predictions_sha256"] != text_sha256(predictions_path)):
        raise ValueError("Declaration prediction run incomplete or inputs/outputs changed")
    registry = load_declaration_registry(registry_path, source_packet_sha256=source_manifest.get("packet_sha256"),
                                         sources_sha256=source_manifest["sources_sha256"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    validate(packet)
    labels = {row["record_id"]: row for row in packet["claims"]}
    sources = {row["record_id"]: row for row in map(json.loads, sources_path.read_text(encoding="utf-8").splitlines())}
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(predictions) != run["prediction_count"] or len(predictions) != source_manifest["source_count"]
            or len({row["record_id"] for row in predictions}) != len(predictions)
            or set(labels) != set(sources) or set(sources) != {row["record_id"] for row in predictions}):
        raise ValueError("Prediction, source, and label IDs or counts differ")
    grounder = DoctrineGrounder(allow_background_axioms=False)
    modes = ("rules", "baseline", "retrieved") if "rules" in run.get("arms", ()) else ("baseline", "retrieved")
    scored = []
    for prediction in predictions:
        rid = prediction["record_id"]
        source = sources[rid]["source_excerpt"]
        if prediction["source_sha256"] != hashlib.sha256(source.encode("utf-8")).hexdigest():
            raise ValueError(f"{rid}: source differs")
        extraction = prediction["extraction"]
        declaration = prediction["declaration"]
        evidence = prediction["evidence_sentence"]
        parents = declaration_parents(registry, declaration) if declaration else []
        if declaration:
            if (extraction["status"] != "candidate"
                    or source[extraction["start"]:extraction["end"]] != extraction["source_span"]
                    or evidence not in source
                    or match_declaration(extraction["candidate_text"], evidence, registry) != declaration
                    or set(prediction["approved_class_terms"]) != set(parents) | {declaration["symbol"]}
                    or set(prediction["arms"]) != set(modes)):
                raise ValueError(f"{rid}: declaration or source evidence does not match")
        elif prediction["arms"]:
            raise ValueError(f"{rid}: generated without a reviewed declaration")
        for mode in modes:
            arm = prediction["arms"].get(mode)
            result = {
                "record_id": rid,
                "paragraph": labels[rid]["paragraph"],
                "mode": mode,
                "label_decision": labels[rid]["decision"],
                "declared_symbol": declaration["symbol"] if declaration else None,
                "evidence_sentence": evidence,
                "predicted_cnl": arm["cnl"] if arm else None,
                "compiled_kif": None,
                "route": "no_declaration" if declaration is None else "pending",
                "error": arm["error"] if arm else None,
                "gold_kif_exact": False,
                "grounder_accepted": None,
                "grounder_reason": None,
                "ontology_restated": None,
                "declared_gate_accepted": None,
                "declared_gate_reason": None,
            }
            if arm:
                if arm["decoder_abstention"]:
                    result.update(route="decoder_abstain", error=arm["decoder_abstention"])
                elif arm["error"]:
                    result["route"] = "error"
                elif not arm["cnl"]:
                    result["route"] = "rules_abstain" if mode == "rules" else "empty_output"
                else:
                    try:
                        kif = CNLCompiler(extra_classes={declaration["symbol"]}).compile(
                            arm["cnl"], require_closed=True, require_argument_kinds=True
                        )
                    except Exception as exc:
                        result.update(route="compile_error", error=f"{type(exc).__name__}: {exc}")
                    else:
                        result["compiled_kif"] = kif
                        result["gold_kif_exact"] = labels[rid]["decision"] == "positive" and kif == labels[rid]["kif"]
                        result["ontology_restated"] = restates_ontology(kif)
                        result["route"] = "ai_label_match" if result["gold_kif_exact"] else "valid_cnl_other_formula"
                        try:
                            assessment = grounder.assess(
                                cnl=arm["cnl"], original_text=source,
                                normalized_text=evidence, linked_text=evidence,
                                subclaim_text=extraction["candidate_text"],
                            )
                            result["grounder_accepted"] = assessment.accepted
                            result["grounder_reason"] = assessment.reason
                        except Exception as exc:
                            result["grounder_reason"] = f"{type(exc).__name__}: {exc}"
                        # Passage-only review: the declaration maps a phrase to a
                        # symbol but never supplies the parent relation.
                        declared = assess_declared_definition(
                            arm["cnl"], passage=source, evidence_sentence=evidence,
                            declaration=declaration, parent_terms=parents,
                        )
                        result["declared_gate_accepted"] = declared.accepted
                        result["declared_gate_reason"] = declared.reason
                        if "gate" in arm and arm["gate"] != {"accepted": declared.accepted, "reason": declared.reason}:
                            raise ValueError(f"{rid}/{mode}: saved gate verdict differs from rescoring")
            scored.append(result)
    summary = {
        "label_status": "ai_reviewed_development_oracle_not_evaluation",
        "source_passages": len(predictions),
        "oracle_declarations_applied": sum(row["declaration"] is not None for row in predictions),
        "memory_retrieved_on_declarations": sum(bool(row["memory"]) for row in predictions if row["declaration"]),
        "by_mode": {},
        "interpretation": "Development-only upper-bound probe: provisional declarations and parent pool were informed by AI-reviewed examples. Exact AI-label match is not independent accuracy or final acceptance.",
    }
    for mode in modes:
        subset = [row for row in scored if row["mode"] == mode]
        summary["by_mode"][mode] = {
            "routes": dict(Counter(row["route"] for row in subset)),
            "exact_ai_label_matches": sum(row["gold_kif_exact"] for row in subset),
            "compiled_other_formulas": sum(row["route"] == "valid_cnl_other_formula" for row in subset),
            "grounder_accepted": sum(row["grounder_accepted"] is True for row in subset),
            "declared_gate_accepted": sum(row["declared_gate_accepted"] is True for row in subset),
            "declared_gate_accepted_ai_label_match": sum(
                row["declared_gate_accepted"] is True and row["gold_kif_exact"] for row in subset),
            "declared_gate_accepted_other_formula": sum(
                row["declared_gate_accepted"] is True and not row["gold_kif_exact"] for row in subset),
            "declared_gate_reasons": dict(Counter(
                row["declared_gate_reason"] or "accepted" for row in subset
                if row["declared_gate_accepted"] is not None)),
        }
    by_mode = {mode: {row["record_id"]: row for row in scored if row["mode"] == mode}
               for mode in ("baseline", "retrieved")}
    summary["changed_output_count"] = sum(
        by_mode["baseline"][rid]["predicted_cnl"] != by_mode["retrieved"][rid]["predicted_cnl"]
        for rid in labels
    )
    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scored, summary = score(args.sources, args.source_manifest, args.registry, args.packet,
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
