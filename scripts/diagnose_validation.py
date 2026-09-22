"""Measure generated CNL quality on the deterministic synthetic validation split.

This split shares templates and vocabulary with training; it is a model
diagnostic, not an independent doctrine benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.compiler import CNLCompiler
from src.eval.model_predictor import load_model_and_tokenizer
from src.training.train import format_prompt, load_training_pairs, split_training_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pairs = load_training_pairs(args.train_file, limit=args.limit)
    training, validation = split_training_pairs(pairs, seed=args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    compiler = CNLCompiler()
    counts = Counter()
    patterns: dict[str, Counter] = defaultdict(Counter)
    records_path = args.output_dir / "predictions.jsonl"
    with records_path.open("w", encoding="utf-8") as output, torch.inference_mode():
        for offset in range(0, len(validation), args.batch_size):
            batch = validation[offset:offset + args.batch_size]
            encoded = tokenizer(
                [format_prompt(pair.nl) for pair in batch],
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            generated = model.generate(**encoded, max_new_tokens=64)
            cnls = [s.strip() for s in tokenizer.batch_decode(generated, skip_special_tokens=True)]
            for pair, cnl in zip(batch, cnls, strict=True):
                error = None
                try:
                    kif = compiler.compile(cnl, require_closed=True)
                except Exception as exc:
                    kif, error = None, f"{type(exc).__name__}: {exc}"
                exact = cnl == pair.cnl
                valid = kif is not None
                expected_kif = pair.kif
                row = {
                    "nl": pair.nl, "pattern": pair.pattern,
                    "expected_cnl": pair.cnl, "generated_cnl": cnl,
                    "expected_kif": expected_kif, "generated_kif": kif,
                    "valid_cnl": valid, "exact_cnl": exact,
                    "exact_kif": kif == expected_kif if expected_kif is not None else None,
                    "compile_error": error,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["total"] += 1
                p = patterns[pair.pattern or "unknown"]
                p["total"] += 1
                if valid:
                    counts["valid_cnl"] += 1
                    p["valid_cnl"] += 1
                if exact:
                    counts["exact_cnl"] += 1
                    p["exact_cnl"] += 1
                if expected_kif is not None and kif == expected_kif:
                    counts["exact_kif"] += 1
                    p["exact_kif"] += 1
    summary = {
        "interpretation": "Synthetic grouped validation split; not an independent doctrine test or semantic entailment check.",
        "train_file": str(args.train_file),
        "train_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "model_path": str(args.model_path),
        "seed": args.seed, "loaded_pairs": len(pairs),
        "train_pairs": len(training), "validation_pairs": len(validation),
        "device": device, "counts": dict(counts),
        "rates": {key: counts[key] / counts["total"] for key in ("valid_cnl", "exact_cnl", "exact_kif")},
        "by_pattern": {key: dict(value) for key, value in sorted(patterns.items())},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
