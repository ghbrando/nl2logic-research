"""Verify and summarize a paired baseline/retrieved-memory development run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import text_sha256


def _read_run(directory: Path, expected_mode: str) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    predictions_path = directory / "predictions.jsonl"
    if (manifest["status"] != "completed" or manifest["memory_mode"] != expected_mode
            or text_sha256(predictions_path) != manifest["predictions_sha256"]):
        raise ValueError(f"Incomplete or changed {expected_mode} predictions")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["prediction_count"] or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError(f"Invalid {expected_mode} prediction IDs or count")
    return manifest, {row["record_id"]: row for row in rows}


def compare(baseline_dir: Path, retrieved_dir: Path) -> dict:
    baseline, left = _read_run(baseline_dir, "none")
    retrieved, right = _read_run(retrieved_dir, "retrieved")
    for key in ("source_manifest_sha256", "sources_sha256", "adapter_sha256", "git_head", "max_new_tokens"):
        if baseline[key] != retrieved[key]:
            raise ValueError(f"Paired runs differ in {key}")
    if set(left) != set(right):
        raise ValueError("Paired prediction IDs differ")
    rows_with_memory = 0
    generation_with_memory = 0
    blocked_before_generation = 0
    changed_outcomes = []
    memory_refs = []
    for rid in left:
        a, b = left[rid], right[rid]
        for key in ("source_sha256", "source_sentence_sha256", "extraction", "source_prompt_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"{rid}: paired inputs or extraction differ in {key}")
        if a["memory"] or a["source_prompt_sha256"] != a["model_prompt_sha256"]:
            raise ValueError(f"{rid}: baseline received memory context")
        if b["memory"]:
            rows_with_memory += 1
            if b["source_prompt_sha256"] == b["model_prompt_sha256"]:
                raise ValueError(f"{rid}: memory prompt did not change")
            memory_refs.append({"record_id": rid, "statements": [item["statement_id"] for item in b["memory"]]})
            if b["decoder_abstention"] == "no lexically supported constrained continuation":
                blocked_before_generation += 1
            elif b["extraction"]["status"] == "candidate" and not b["extraction"]["gate_reasons"]:
                generation_with_memory += 1
        outputs = ("constrained_cnl", "decoder_abstention", "error")
        if any(a[key] != b[key] for key in outputs):
            changed_outcomes.append(rid)
    return {
        "source_passages": len(left),
        "adapter_sha256": baseline["adapter_sha256"],
        "git_head": baseline["git_head"],
        "baseline_predictions_sha256": baseline["predictions_sha256"],
        "retrieved_predictions_sha256": retrieved["predictions_sha256"],
        "rows_with_retrieved_memory": rows_with_memory,
        "memory_blocked_before_generation": blocked_before_generation,
        "memory_entered_generation": generation_with_memory,
        "changed_output_count": len(changed_outcomes),
        "changed_output_ids": changed_outcomes,
        "memory_refs": memory_refs,
        "interpretation": "This run cannot estimate an attention benefit: every memory-augmented candidate was blocked by the unchanged source-only grammar before generation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--retrieved-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.baseline_dir, args.retrieved_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
