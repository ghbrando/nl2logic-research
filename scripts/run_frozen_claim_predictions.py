"""Run source-only, CPU inference on a frozen partial-claim source selection.

This script intentionally has no label-packet argument. Scoring is a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.comparison_predictors import RulesPredictor
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler
from src.training.train import format_prompt, generate_cnl_outputs


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_sources(selected_path: Path, selection_manifest_path: Path) -> list[dict]:
    manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    expected = manifest["frozen_artifact_sha256"]["selected_passages_sha256"]
    if text_sha256(selected_path) != expected:
        raise ValueError("Selected source passages differ from frozen manifest")
    rows = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["selected_count"] or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Selected source passage count or IDs changed")
    if any(row.get("decision") is not None or row.get("status") != "source_selected_unreviewed" for row in rows):
        raise ValueError("Prediction inputs contain labels or reviewed rows")
    return rows


def predict(
    selected_path: Path,
    selection_manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    max_new_tokens: int = 64,
) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    rows = load_sources(selected_path, selection_manifest_path)
    import torch

    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("This sealed inference runner requires a CPU-only model")
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.num_beams = 1
    rules = RulesPredictor(allow_background_axioms=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = model_path / "adapter_model.safetensors"
    run_manifest = {
        "status": "running",
        "protocol": "Full source paragraph to rules and raw greedy seq2seq; raw bypasses the unsupported-input gate. Gate reasons are recorded without generating a constrained prediction. No labels loaded.",
        "selected_path": str(selected_path),
        "selected_sha256": text_sha256(selected_path),
        "source_selection_manifest_sha256": text_sha256(selection_manifest_path),
        "model_path": str(model_path),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "device": "cpu",
        "generation": {"do_sample": False, "num_beams": 1},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "prediction_count": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            source = row["source_excerpt"]
            gate_reasons = CNLSampler._unsupported_reasons(source)
            start = perf_counter()
            rules_cnl, rules_error = None, None
            try:
                rules_cnl = rules(source)
            except Exception as exc:
                rules_error = f"{type(exc).__name__}: {exc}"
            raw_cnl, raw_error = None, None
            try:
                with torch.inference_mode():
                    output = generate_cnl_outputs(
                        model, tokenizer, [format_prompt(source)], max_new_tokens=max_new_tokens
                    )
                raw_cnl = output[0] if output else None
            except Exception as exc:
                raw_error = f"{type(exc).__name__}: {exc}"
            result = {
                "record_id": row["record_id"],
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "gate_reasons": gate_reasons,
                "rules_cnl": rules_cnl,
                "rules_error": rules_error,
                "raw_cnl": raw_cnl,
                "raw_error": raw_error,
                "latency_seconds": perf_counter() - start,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {row['record_id']} raw={raw_cnl!r}", flush=True)
    run_manifest["prediction_count"] = len(rows)
    run_manifest["predictions_sha256"] = text_sha256(predictions_path)
    run_manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    predict(args.selected, args.selection_manifest, args.model_path, args.output_dir,
            max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
