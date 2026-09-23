"""Run source-only definition-head extraction and constrained CPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import text_sha256
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.preprocessing.partial_claims import extract_paragraph_candidate
from src.training.train import format_prompt


def load_sources(source_path: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if text_sha256(source_path) != manifest["sources_sha256"]:
        raise ValueError("Development source projection changed")
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["source_count"] or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Development source count or IDs changed")
    if any("decision" in row or "cnl" in row or "kif" in row for row in rows):
        raise ValueError("Inference source rows contain labels or formal claims")
    return rows


def predict(source_path: Path, source_manifest_path: Path, model_path: Path,
            output_dir: Path, *, max_new_tokens: int = 64) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    rows = load_sources(source_path, source_manifest_path)
    extracted = [(row, *extract_paragraph_candidate(row["source_excerpt"], row["source_sentence"])) for row in rows]
    eligible = [(row, candidate) for row, candidate, _ in extracted if candidate.status == "candidate" and not candidate.gate_reasons]
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only development container unexpectedly has CUDA access")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path) if eligible else (None, None)
    if model is not None and any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Development probe requires a CPU-only model")
    if model is not None:
        model.eval()
    sampler = CNLSampler(model, tokenizer, allow_background_axioms=False) if eligible else None
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "protocol": "Unlabeled chapter 1 development paragraph -> source-preserving sentence scan -> first conservative definition head -> production gate -> constrained decoder. No labels loaded; no claim accepted by this script.",
        "source_manifest_sha256": text_sha256(source_manifest_path),
        "sources_sha256": text_sha256(source_path),
        "adapter_sha256": hashlib.sha256((model_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "device": "cpu",
        "generation": {"decoder": "CNLSampler", "num_beams": 4},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_count": len(rows),
        "eligible_count": len(eligible),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row, candidate, screened in extracted:
            cnl, error, decoder_abstention = None, None, None
            if candidate.status == "candidate" and not candidate.gate_reasons:
                try:
                    cnl = sampler.sample(format_prompt(candidate.candidate_text), max_tokens=max_new_tokens)
                except UnsupportedInputError as exc:
                    decoder_abstention = str(exc)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            result = {
                "record_id": row["record_id"],
                "paragraph": row["paragraph"],
                "source_sha256": hashlib.sha256(row["source_excerpt"].encode("utf-8")).hexdigest(),
                "source_sentence_sha256": hashlib.sha256(row["source_sentence"].encode("utf-8")).hexdigest(),
                "extraction": asdict(candidate),
                "screened_sentences": screened,
                "constrained_cnl": cnl,
                "decoder_abstention": decoder_abstention,
                "error": error,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{row['paragraph']} {candidate.status} {candidate.reason or candidate.candidate_text!r} cnl={cnl!r} abstention={decoder_abstention!r} error={error!r}", flush=True)
    manifest["prediction_count"] = len(rows)
    manifest["predictions_sha256"] = text_sha256(prediction_path)
    manifest["status"] = "completed"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    predict(args.sources, args.source_manifest, args.model_path, args.output_dir,
            max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
