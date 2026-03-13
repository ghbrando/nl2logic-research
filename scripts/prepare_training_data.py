from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.compiler.compiler import CNLCompiler
from src.data.generate_pairs import (
    _CLASSES_PATH,
    _RELATIONS_PATH,
    count_pairs_by_pattern,
    gen_binary_pairs,
    gen_conditional_every_pairs,
    gen_conditional_if_pairs,
    gen_existential_pairs,
    gen_instance_pairs,
    gen_negation_pairs,
    gen_nary_pairs,
    gen_subclass_pairs,
    load_classes,
    load_relations,
    select_output_pairs,
)

DEFAULT_LIMIT = 50_000
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "training_pairs" / "train_balanced_50k.jsonl"
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SAMPLE_SEED = 42


@dataclass(frozen=True)
class PreparationResult:
    output_path: Path
    pairs: list[dict]
    written_counts: dict[str, int]
    generated_counts: dict[str, int]
    balanced_cap: int
    samples_by_pattern: dict[str, list[dict]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced NL->CNL training artifact for Stage 1 fine-tuning."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"Cap total written pairs (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Prepared artifact output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        metavar="N",
        help=f"Number of inspection samples to print per pattern (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        metavar="N",
        help=f"Seed for deterministic sample inspection (default: {DEFAULT_SAMPLE_SEED})",
    )
    return parser.parse_args(argv)


def build_generator_batches(
    classes: list[dict],
    relations: list[dict],
    compiler: CNLCompiler,
) -> list[tuple[str, list[dict]]]:
    return [
        ("instance", gen_instance_pairs(classes, compiler)),
        ("subclass", gen_subclass_pairs(classes, compiler)),
        ("binary", gen_binary_pairs(relations, compiler)),
        ("nary", gen_nary_pairs(relations, compiler)),
        ("conditional", gen_conditional_every_pairs(relations, compiler)),
        ("conditional", gen_conditional_if_pairs(relations, compiler)),
        ("existential", gen_existential_pairs(classes, compiler)),
        ("negation", gen_negation_pairs(classes, relations, compiler)),
    ]


def _write_pairs(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def build_samples_by_pattern(
    pairs: list[dict],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    pattern_order: list[str] | None = None,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for pair in pairs:
        grouped.setdefault(pair["pattern"], []).append(pair)

    ordered_patterns = pattern_order or list(grouped)
    rng = random.Random(sample_seed)
    samples: dict[str, list[dict]] = {}

    for pattern in ordered_patterns:
        bucket = grouped.get(pattern, [])
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        samples[pattern] = shuffled[:sample_size]

    return samples


def prepare_training_data(
    *,
    limit: int = DEFAULT_LIMIT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    generator_batches: list[tuple[str, list[dict]]] | None = None,
) -> PreparationResult:
    if generator_batches is None:
        classes = load_classes(_CLASSES_PATH)
        relations = load_relations(_RELATIONS_PATH)
        compiler = CNLCompiler()
        generator_batches = build_generator_batches(classes, relations, compiler)

    all_pairs: list[dict] = []
    generated_counts: dict[str, int] = {}

    for label, batch in generator_batches:
        generated_counts[label] = generated_counts.get(label, 0) + len(batch)
        all_pairs.extend(batch)

    output_pairs = select_output_pairs(all_pairs, limit=limit, balanced=True)
    written_counts = count_pairs_by_pattern(output_pairs)
    balanced_cap = limit // len(generated_counts) if generated_counts else 0
    samples_by_pattern = build_samples_by_pattern(
        output_pairs,
        sample_size=sample_size,
        sample_seed=sample_seed,
        pattern_order=list(generated_counts),
    )

    _write_pairs(output_path, output_pairs)

    return PreparationResult(
        output_path=output_path,
        pairs=output_pairs,
        written_counts=written_counts,
        generated_counts=generated_counts,
        balanced_cap=balanced_cap,
        samples_by_pattern=samples_by_pattern,
    )


def print_preparation_report(result: PreparationResult) -> None:
    print("Prepared balanced training artifact:")
    print(f"  Output:               {result.output_path}")
    print(f"  Total pairs written:  {len(result.pairs):,}")
    print(f"  Balanced cap/pattern: {result.balanced_cap:,}")
    print("  Per-pattern counts:")
    for pattern in result.generated_counts:
        print(f"    {pattern:15s}: {result.written_counts.get(pattern, 0):>7,}")

    print("\nInspection samples:")
    for pattern in result.generated_counts:
        print(f"[{pattern}]")
        samples = result.samples_by_pattern.get(pattern, [])
        if not samples:
            print("  <no samples written>")
            continue
        for index, sample in enumerate(samples, start=1):
            print(f"  {index}. NL:  {sample['nl']}")
            print(f"     CNL: {sample['cnl']}")
            print(f"     KIF: {sample['kif']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = prepare_training_data(
        limit=args.limit,
        output_path=args.output,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    print_preparation_report(result)


if __name__ == "__main__":
    main()
