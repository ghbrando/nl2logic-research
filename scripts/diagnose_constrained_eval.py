from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.evaluate import DEFAULT_GOLD_PATH, load_gold_pairs
from src.eval.model_predictor import DEFAULT_MODEL_PATH, load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, build_xgrammar_grammar
from src.training.train import format_prompt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether constrained evaluation time is spent in grammar "
            "build, xgrammar init, or the first constrained decode."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Gold JSONL path (default: {DEFAULT_GOLD_PATH})",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="0-based gold example index to probe (default: 0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens for the constrained sample (default: 64)",
    )
    parser.add_argument(
        "--grammar-mode",
        choices=["full", "control"],
        default="full",
        help=(
            "Grammar to use for the constrained sample: full SUMO vocabulary "
            "or a single-literal control grammar (default: full)"
        ),
    )
    return parser.parse_args(argv)


def _read_vocab_counts() -> tuple[int, int]:
    classes_path = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
    relations_path = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"

    with open(classes_path, encoding="utf-8") as handle:
        class_count = sum(1 for line in handle if line.strip())
    with open(relations_path, encoding="utf-8") as handle:
        relation_count = sum(1 for line in handle if line.strip())

    return class_count, relation_count


def _print_step(label: str) -> float:
    print(label, flush=True)
    return time.perf_counter()


def _finish_step(start: float, detail: str = "") -> float:
    elapsed = time.perf_counter() - start
    suffix = f" {detail}" if detail else ""
    print(f"  done in {elapsed:.2f}s{suffix}", flush=True)
    return elapsed


def build_control_grammar(cnl: str) -> str:
    """
    Build the smallest possible constrained grammar for a single target CNL string.

    This is a control experiment: if the backend is healthy, decoding against a
    one-literal grammar should complete quickly.
    """
    return f"root ::= {json.dumps(cnl.strip())}"


def select_probe_grammar(*, gold_cnl: str, grammar_mode: str) -> tuple[str, str]:
    if grammar_mode == "full":
        return build_xgrammar_grammar(), "full closed-vocabulary grammar"
    if grammar_mode == "control":
        return build_control_grammar(gold_cnl), "single-literal control grammar"
    raise ValueError(f"Unknown grammar mode: {grammar_mode}")


def run_probe(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
    pair_index: int = 0,
    max_new_tokens: int = 64,
    grammar_mode: str = "full",
) -> None:
    pairs = load_gold_pairs(gold_path)
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} is out of range for {len(pairs)} gold pairs")

    pair = pairs[pair_index]
    prompt = format_prompt(pair.nl)
    class_count, relation_count = _read_vocab_counts()

    print(f"Gold pairs loaded: {len(pairs)}", flush=True)
    print(f"Probe pair index: {pair_index}", flush=True)
    print(f"Pattern: {pair.pattern}", flush=True)
    print(f"NL: {pair.nl}", flush=True)
    print(f"Gold CNL: {pair.cnl}", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(
        f"Vocabulary size: {class_count} classes, {relation_count} relations",
        flush=True,
    )
    print(f"Grammar mode: {grammar_mode}", flush=True)

    start = _print_step("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    _finish_step(start)

    start = _print_step("Building constrained grammar string...")
    grammar, grammar_label = select_probe_grammar(gold_cnl=pair.cnl, grammar_mode=grammar_mode)
    _finish_step(start, detail=f"({grammar_label}; {len(grammar):,} chars)")

    start = _print_step("Constructing sampler...")
    sampler = CNLSampler(model, tokenizer, grammar_str=grammar)
    _finish_step(start)

    start = _print_step("Initializing xgrammar compiler / logits processor...")
    sampler._get_xgrammar_logits_processors()
    _finish_step(start)

    start = _print_step("Running first constrained sample...")
    prediction = sampler.sample(prompt, max_tokens=max_new_tokens)
    _finish_step(start)

    print("Prediction:", flush=True)
    print(f"  {prediction}", flush=True)

    start = _print_step("Compiling prediction...")
    kif = sampler.validate_output(prediction)
    _finish_step(start)

    print("Compiled KIF:", flush=True)
    print(f"  {kif}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_probe(
        model_path=args.model_path,
        gold_path=args.gold_path,
        pair_index=args.pair_index,
        max_new_tokens=args.max_new_tokens,
        grammar_mode=args.grammar_mode,
    )


if __name__ == "__main__":
    main()




