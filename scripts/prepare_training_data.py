from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.compiler.compiler import CNLCompiler
from src.data.generate_pairs import (
    _CLASSES_PATH,
    _DOCTRINE_KIF_PATH,
    _RELATIONS_PATH,
    count_pairs_by_pattern,
    gen_binary_pairs,
    gen_conditional_every_pairs,
    gen_conditional_if_pairs,
    gen_doctrine_subclass_pairs,
    gen_existential_pairs,
    gen_instance_pairs,
    gen_negation_pairs,
    gen_nary_pairs,
    gen_subclass_pairs,
    load_classes,
    load_relations,
)
from src.eval.evaluate import DEFAULT_GOLD_PATH
from src.training.train import PATTERN_COVERAGE_SENTENCES

DEFAULT_LIMIT = 50_000
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "training_pairs" / "train_balanced_50k.jsonl"
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SAMPLE_SEED = 42
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MILITARY_FOCUS_CLASSES = {
    # Base SUMO military classes
    "Area",
    "Artifact",
    "Attack",
    "AutonomousAgent",
    "Battle",
    "Communication",
    "MilitaryOrganization",
    "MilitaryProcess",
    "MilitaryUnit",
    "Order",
    "Organization",
    "Plan",
    "Process",
    "Region",
    "Transportation",
    "Weapon",
    # Doctrine domain extensions (doctrine_domain.kif)
    "CombatInformation",
    "DoctrinalTask",
    "GeospatialIntelligence",
    "HumanIntelligence",
    "InformationCollection",
    "IntelligenceDiscipline",
    "IntelligenceDissemination",
    "IntelligenceEnterprise",
    "IntelligenceProcess",
    "IntelligenceProduct",
    "IntelligenceProfessional",
    "IntelligenceWarfightingFunction",
    "IntelligenceWarfightingFunctionTask",
    "OperationalEnvironment",
    "SignalsIntelligence",
    "TacticalCommander",
    "ThreatCourseOfAction",
    "WarfightingFunction",
}
MILITARY_FOCUS_RELATIONS = {
    "agent",
    "between",
    "destination",
    "instrument",
    "located",
    "orientation",
    "origin",
    "patient",
}


@dataclass(frozen=True)
class PreparationResult:
    output_path: Path
    pairs: list[dict]
    written_counts: dict[str, int]
    generated_counts: dict[str, int]
    balanced_cap: int
    samples_by_pattern: dict[str, list[dict]]
    excluded_counts: dict[str, int]
    focused_counts: dict[str, int]


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
    parser.add_argument(
        "--allow-gold-cnl-overlap",
        action="store_true",
        help="Keep training paraphrases whose CNL matches the gold eval set while still excluding exact gold NL and probe NL overlaps.",
    )
    return parser.parse_args(argv)


def build_generator_batches(
    classes: list[dict],
    relations: list[dict],
    compiler: CNLCompiler,
) -> list[tuple[str, list[dict]]]:
    return [
        ("instance", gen_instance_pairs(classes, compiler, include_doc_templates=False)),
        ("subclass", gen_subclass_pairs(classes, compiler, include_doc_templates=False)),
        ("subclass", gen_doctrine_subclass_pairs(_DOCTRINE_KIF_PATH, compiler)),
        ("binary", gen_binary_pairs(relations, compiler, include_doc_templates=False)),
        ("nary", gen_nary_pairs(relations, compiler, include_doc_templates=False)),
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


def is_military_focus_pair(
    pair: dict,
    *,
    focus_classes: set[str] = MILITARY_FOCUS_CLASSES,
    focus_relations: set[str] = MILITARY_FOCUS_RELATIONS,
) -> bool:
    tokens = set(_TOKEN_RE.findall(pair.get("cnl", "")))
    return bool(tokens & focus_classes or tokens & focus_relations)


def select_prepared_pairs(
    pairs: list[dict],
    *,
    limit: int,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[list[dict], dict[str, int]]:
    if limit <= 0:
        selected = list(pairs)
        return selected, count_pairs_by_pattern([pair for pair in selected if is_military_focus_pair(pair)])

    grouped: dict[str, list[dict]] = {}
    for pair in pairs:
        grouped.setdefault(pair["pattern"], []).append(pair)

    if not grouped:
        return [], {}

    per_pattern_cap = limit // len(grouped)
    rng = random.Random(seed)
    selected: list[dict] = []
    focused_counts: dict[str, int] = {}

    for pattern, bucket in grouped.items():
        focused = [pair for pair in bucket if is_military_focus_pair(pair)]
        background = [pair for pair in bucket if not is_military_focus_pair(pair)]
        rng.shuffle(focused)
        rng.shuffle(background)

        chosen = focused[:per_pattern_cap]
        if len(chosen) < per_pattern_cap:
            chosen.extend(background[: per_pattern_cap - len(chosen)])

        selected.extend(chosen)
        focused_counts[pattern] = sum(1 for pair in chosen if is_military_focus_pair(pair))

    return selected, focused_counts


def _load_gold_overlap_sets(gold_path: Path = DEFAULT_GOLD_PATH) -> tuple[set[str], set[str]]:
    with open(gold_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return (
        {record["nl"] for record in records if "nl" in record},
        {record["cnl"] for record in records if "cnl" in record},
    )


def exclude_eval_overlaps(
    pairs: list[dict],
    *,
    excluded_nls: set[str] | None = None,
    excluded_cnls: set[str] | None = None,
    exclude_gold_cnl_overlap: bool = True,
    probe_sentences: list[tuple[str, str]] = PATTERN_COVERAGE_SENTENCES,
) -> tuple[list[dict], dict[str, int]]:
    if excluded_nls is None or excluded_cnls is None:
        gold_nls, gold_cnls = _load_gold_overlap_sets()
        if excluded_nls is None:
            excluded_nls = gold_nls
        if excluded_cnls is None:
            excluded_cnls = gold_cnls if exclude_gold_cnl_overlap else set()

    excluded_probe_nls = {nl for _, nl in probe_sentences}

    filtered: list[dict] = []
    excluded_counts = {
        "gold_nl": 0,
        "gold_cnl": 0,
        "probe_nl": 0,
    }

    for pair in pairs:
        if pair["nl"] in excluded_probe_nls:
            excluded_counts["probe_nl"] += 1
            continue
        if pair["nl"] in excluded_nls:
            excluded_counts["gold_nl"] += 1
            continue
        if pair["cnl"] in excluded_cnls:
            excluded_counts["gold_cnl"] += 1
            continue
        filtered.append(pair)

    return filtered, excluded_counts


def prepare_training_data(
    *,
    limit: int = DEFAULT_LIMIT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    generator_batches: list[tuple[str, list[dict]]] | None = None,
    excluded_nls: set[str] | None = None,
    excluded_cnls: set[str] | None = None,
    exclude_gold_cnl_overlap: bool = True,
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

    all_pairs, excluded_counts = exclude_eval_overlaps(
        all_pairs,
        excluded_nls=excluded_nls,
        excluded_cnls=excluded_cnls,
        exclude_gold_cnl_overlap=exclude_gold_cnl_overlap,
    )
    output_pairs, focused_counts = select_prepared_pairs(
        all_pairs,
        limit=limit,
        seed=sample_seed,
    )
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
        excluded_counts=excluded_counts,
        focused_counts=focused_counts,
    )


def print_preparation_report(result: PreparationResult) -> None:
    print("Prepared balanced training artifact:")
    print(f"  Output:               {result.output_path}")
    print(f"  Total pairs written:  {len(result.pairs):,}")
    print(f"  Balanced cap/pattern: {result.balanced_cap:,}")
    print(f"  Excluded gold NL:     {result.excluded_counts['gold_nl']:,}")
    print(f"  Excluded gold CNL:    {result.excluded_counts['gold_cnl']:,}")
    print(f"  Excluded probe NL:    {result.excluded_counts['probe_nl']:,}")
    print("  Per-pattern counts:")
    for pattern in result.generated_counts:
        print(f"    {pattern:15s}: {result.written_counts.get(pattern, 0):>7,}")
    print("  Military-focus counts:")
    for pattern in result.generated_counts:
        print(f"    {pattern:15s}: {result.focused_counts.get(pattern, 0):>7,}")

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
        exclude_gold_cnl_overlap=not args.allow_gold_cnl_overlap,
    )
    print_preparation_report(result)


if __name__ == "__main__":
    main()
