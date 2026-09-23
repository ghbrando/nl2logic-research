"""Run the production input gate and constrained decoder on frozen source only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import load_sources, text_sha256
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.training.train import format_prompt


def predict(selected_path: Path, selection_manifest_path: Path, model_path: Path,
            output_dir: Path, *, max_new_tokens: int = 64) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    rows = load_sources(selected_path, selection_manifest_path)
    gated = [(row, CNLSampler._unsupported_reasons(row["source_excerpt"])) for row in rows]
    eligible = [row for row, reasons in gated if not reasons]
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only audit container unexpectedly has CUDA access")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path) if eligible else (None, None)
    if model is not None and any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("This guarded runner requires a CPU-only model")
    if model is not None:
        model.eval()
    sampler = CNLSampler(model, tokenizer, allow_background_axioms=False) if eligible else None
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "protocol": "Full source paragraph -> production unsupported-input gate -> constrained decoder only for eligible passages. No labels loaded.",
        "selected_sha256": text_sha256(selected_path),
        "source_selection_manifest_sha256": text_sha256(selection_manifest_path),
        "adapter_sha256": hashlib.sha256((model_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "device": "cpu",
        "generation": {"decoder": "CNLSampler", "num_beams": 4},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "eligible_count": len(eligible),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_path = output_dir / "guarded_predictions.jsonl"
    with output_path.open("x", encoding="utf-8") as handle:
        for row, reasons in gated:
            cnl, error = None, None
            if not reasons:
                try:
                    cnl = sampler.sample(format_prompt(row["source_excerpt"]), max_tokens=max_new_tokens)
                except UnsupportedInputError as exc:
                    reasons = [str(exc)]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            result = {
                "record_id": row["record_id"],
                "source_sha256": hashlib.sha256(row["source_excerpt"].encode("utf-8")).hexdigest(),
                "gate_reasons": reasons,
                "constrained_cnl": cnl,
                "error": error,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{row['record_id']} constrained={cnl!r} error={error!r}", flush=True)
    manifest["prediction_count"] = len(rows)
    manifest["predictions_sha256"] = text_sha256(output_path)
    manifest["status"] = "completed"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
