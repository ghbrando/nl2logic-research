"""Compare rules, raw seq2seq, and constrained seq2seq on identical inputs."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.comparison import evaluate_method, fingerprint, read_benchmark, summarize, training_overlap
from src.eval.comparison_predictors import RulesPredictor, Seq2SeqPredictor
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import _DEFAULT_CONSTRAINED_NUM_BEAMS, _DEFAULT_CONSTRAINED_LENGTH_PENALTY


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", help="Repeat for each benchmark JSONL")
    parser.add_argument("--methods", nargs="+", choices=["rules", "raw", "constrained"], default=["rules", "raw", "constrained"])
    parser.add_argument("--model-path", type=Path, help="Same checkpoint for both model arms; no oracle fallback")
    parser.add_argument("--training-data", type=Path, action="append", help="Repeat for actual checkpoint training files; overlapping texts are unscored")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    directory = ROOT / "data/benchmarks/fm2-0"
    paths = args.benchmark or [directory / "seed_positive.jsonl", directory / "seed_abstain.jsonl"]
    training_paths = args.training_data or []
    try:
        records = read_benchmark(paths)
        overlaps = training_overlap(records, training_paths)
        output = args.output_dir or ROOT / "results/comparisons" / uuid4().hex
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot prepare comparison: {exc}\n")
    packages = {}
    for name in ("lark", "torch", "transformers", "peft"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    code_paths = [ROOT / "src/eval/comparison.py", ROOT / "src/eval/comparison_predictors.py",
                  ROOT / "src/fsm/cnl_fsm.py", ROOT / "src/ingest/grounding.py",
                  ROOT / "src/compiler/compiler.py", ROOT / "src/compiler/cnl.lark",
                  ROOT / "src/ontology/vocab.py"]
    knowledge_paths = [ROOT / "data/training_pairs/sumo_classes.jsonl", ROOT / "data/training_pairs/sumo_relations.jsonl",
                       ROOT / "data/ontology/doctrine_domain.kif"]
    manifest = {
        "benchmark_files": [fingerprint(p) for p in paths],
        "training_files": [fingerprint(p) for p in training_paths],
        "training_text_overlap": overlaps,
        "leakage_check": "normalized_exact_text_only" if training_paths else "not_checked_training_files_not_supplied",
        "code": [fingerprint(p) for p in code_paths], "knowledge": [fingerprint(p) for p in knowledge_paths],
        "python": platform.python_version(), "packages": packages,
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "methods": list(dict.fromkeys(args.methods)), "max_tokens": args.max_tokens,
        "generation": {"num_beams": _DEFAULT_CONSTRAINED_NUM_BEAMS,
                       "length_penalty": _DEFAULT_CONSTRAINED_LENGTH_PENALTY,
                       "do_sample": False, "early_stopping": True, "renormalize_logits": True},
        "protocol": "Original passage -> shared unsupported-input gate -> predictor -> shared compiler -> shared grounding. No normalization/decomposition.",
        "interpretation": "Diagnostic benchmark, not a blind held-out test. Rules and grounding share existing ontology mappings; drafts never contribute to scored metrics.",
    }
    if args.model_path and args.model_path.is_dir():
        manifest["checkpoint_files"] = [fingerprint(p) for p in sorted(args.model_path.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "benchmark_snapshot.jsonl", records)
    predictors = {}
    unavailable = {}
    if "rules" in args.methods:
        predictors["rules"] = RulesPredictor()
    requested_models = [m for m in dict.fromkeys(args.methods) if m != "rules"]
    if requested_models:
        try:
            if args.model_path is None:
                raise ValueError("No --model-path supplied")
            model, tokenizer = load_model_and_tokenizer(args.model_path)
            if hasattr(model, "generation_config"):
                model.generation_config.do_sample = False
            for method in requested_models:
                predictors[method] = Seq2SeqPredictor(model, tokenizer, constrained=method == "constrained", max_tokens=args.max_tokens)
        except Exception as exc:
            unavailable = {method: f"{type(exc).__name__}: {exc}" for method in requested_models}
    reports, review_queue = {}, []
    for method in dict.fromkeys(args.methods):
        if method in unavailable:
            reports[method] = {"status": "unavailable", "reason": unavailable[method], "metrics": None}
            continue
        results = evaluate_method(records, predictors[method], overlaps=overlaps)
        write_jsonl(output / f"{method}.jsonl", results)
        status = "completed_with_errors" if any(r["route"] == "error" for r in results) else "completed"
        reports[method] = {
            "status": status, "metrics": summarize(results),
            "by_document": {doc: summarize([r for r in results if r["doc_id"] == doc])
                            for doc in sorted({r["doc_id"] for r in results}, key=str)},
            "by_phenomenon": {tag: summarize([r for r in results if tag in r["phenomena"]])
                              for tag in sorted({tag for r in results for tag in r["phenomena"]})},
        }
        review_queue.extend({"method": method, **r, "review_decision": None, "review_seconds": None,
                             "changes_downstream_answer": None, "affected_query": None}
                            for r in results if r["route"] == "review" or r["accepted_mismatch"] or
                            (not r["scored"] and r["route"] == "accepted"))
    complete = all(r["status"] == "completed" for r in reports.values())
    write_json(output / "summary.json", {"requested_runs_complete": complete, "methods": reports})
    write_jsonl(output / "review_queue.jsonl", review_queue)
    print("Method       Status                  Scored  Accepted exact precision  Accepted coverage  Novel coverage")
    for method, report in reports.items():
        metrics = report.get("metrics")
        if metrics is None:
            print(f"{method:<12} unavailable: {report['reason']}")
        else:
            pct = lambda value: "n/a" if value is None else f"{100 * value:.1f}%"
            print(f"{method:<12} {report['status']:<23} {metrics['scored']:>6}  {pct(metrics['accepted_exact_precision']):>24}  {pct(metrics['accepted_coverage']):>17}  {pct(metrics['accepted_novel_coverage']):>14}")
    print("Novel coverage excludes accepts that only restate claims already asserted in doctrine_domain.kif.")
    print(f"\nDiagnostic exact-match metrics; drafts excluded. Training overlap check: {manifest['leakage_check']}.")
    print(f"Artifacts: {output.resolve()}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
