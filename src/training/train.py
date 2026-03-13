"""
Stage 1 fine-tuning script for NL2Logic.

Trains ``google/flan-t5-small`` with LoRA adapters on NL -> CNL pairs, then
runs a small held-out sanity check that compiles generated CNL back to KIF.

This module keeps HuggingFace/PEFT imports lazy so it can still be imported
and unit-tested on the local Windows dev environment where those packages are
not installed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from src.compiler.compiler import CNLCompiler

DEFAULT_MODEL_NAME = "google/flan-t5-small"
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "training_pairs" / "nl_cnl_pairs.jsonl"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "models" / "flan-t5-small-cnl"
DEFAULT_LIMIT = 50_000
DEFAULT_EPOCHS = 3
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_SOURCE_MAX_LENGTH = 256
DEFAULT_TARGET_MAX_LENGTH = 128
DEFAULT_SANITY_CHECK_SAMPLES = 5
PROMPT_PREFIX = "translate to CNL: "
PATTERN_COVERAGE_SENTENCES = [
    ("instance", "Something is an instance of Process."),
    ("subclass", "Weapon is a type of Artifact."),
    ("binary", "The agent relation holds between a Process and an Agent."),
    ("conditional", "Every Process is in the agent relation with an Agent."),
    ("existential", "There exists an instance of MilitaryProcess."),
    ("nary", "The between relation holds among three entities."),
    ("negation", "Something is not an instance of Process."),
]


@dataclass(frozen=True)
class TrainingPair:
    nl: str
    cnl: str
    kif: str | None = None
    pattern: str | None = None


class TokenizedPairDataset:
    """Simple Trainer-compatible dataset backed by pre-tokenized records."""

    def __init__(self, records: list[dict[str, list[int]]]):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._records[index]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune google/flan-t5-small with LoRA for NL -> CNL."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Training-pair JSONL path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base seq2seq model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help="Cap loaded training pairs (0 = no limit, default: 50000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        metavar="N",
        help="Number of fine-tuning epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help="Per-device train/eval batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        metavar="LR",
        help="Learning rate for adapter training (default: 5e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help="Random seed for the train/validation split (default: 42)",
    )
    parser.add_argument(
        "--source-max-length",
        type=int,
        default=DEFAULT_SOURCE_MAX_LENGTH,
        metavar="N",
        help="Maximum source token length (default: 256)",
    )
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=DEFAULT_TARGET_MAX_LENGTH,
        metavar="N",
        help="Maximum target token length (default: 128)",
    )
    return parser.parse_args(argv)


def format_prompt(nl: str) -> str:
    return f"{PROMPT_PREFIX}{nl.strip()}"


def load_training_pairs(
    path: Path = DEFAULT_DATA_PATH,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[TrainingPair]:
    if not path.exists():
        raise FileNotFoundError(f"Training-pair file not found: {path}")

    pairs: list[TrainingPair] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit > 0 and len(pairs) >= limit:
                break

            if not line.strip():
                continue

            record = json.loads(line)
            if "nl" not in record or "cnl" not in record:
                raise ValueError(
                    f"Line {line_number} in {path} must contain 'nl' and 'cnl' fields."
                )

            pairs.append(
                TrainingPair(
                    nl=record["nl"],
                    cnl=record["cnl"],
                    kif=record.get("kif"),
                    pattern=record.get("pattern"),
                )
            )

    if not pairs:
        raise ValueError(f"No training pairs loaded from {path}")

    return pairs


def split_training_pairs(
    pairs: list[TrainingPair],
    *,
    validation_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    if len(pairs) < 2:
        raise ValueError("Need at least 2 training pairs for a 90/10 split.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    validation_size = int(round(len(shuffled) * validation_fraction))
    validation_size = max(1, validation_size)
    validation_size = min(len(shuffled) - 1, validation_size)

    validation_pairs = shuffled[:validation_size]
    training_pairs = shuffled[validation_size:]
    return training_pairs, validation_pairs


def _tokenize_targets(
    tokenizer: Any,
    targets: list[str],
    *,
    max_length: int,
) -> dict[str, list[list[int]]]:
    try:
        return tokenizer(
            text_target=targets,
            max_length=max_length,
            truncation=True,
        )
    except TypeError:
        if hasattr(tokenizer, "as_target_tokenizer"):
            with tokenizer.as_target_tokenizer():
                return tokenizer(
                    targets,
                    max_length=max_length,
                    truncation=True,
                )
        return tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
        )


def build_tokenized_records(
    pairs: list[TrainingPair],
    tokenizer: Any,
    *,
    source_max_length: int = DEFAULT_SOURCE_MAX_LENGTH,
    target_max_length: int = DEFAULT_TARGET_MAX_LENGTH,
) -> list[dict[str, list[int]]]:
    prompts = [format_prompt(pair.nl) for pair in pairs]
    targets = [pair.cnl for pair in pairs]

    source_tokens = tokenizer(
        prompts,
        max_length=source_max_length,
        truncation=True,
    )
    target_tokens = _tokenize_targets(
        tokenizer,
        targets,
        max_length=target_max_length,
    )

    attention_masks = source_tokens.get("attention_mask")
    if attention_masks is None:
        attention_masks = [[1] * len(ids) for ids in source_tokens["input_ids"]]

    records: list[dict[str, list[int]]] = []
    for input_ids, attention_mask, labels in zip(
        source_tokens["input_ids"],
        attention_masks,
        target_tokens["input_ids"],
        strict=True,
    ):
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return records


def build_datasets(
    training_pairs: list[TrainingPair],
    validation_pairs: list[TrainingPair],
    tokenizer: Any,
    *,
    source_max_length: int = DEFAULT_SOURCE_MAX_LENGTH,
    target_max_length: int = DEFAULT_TARGET_MAX_LENGTH,
) -> tuple[TokenizedPairDataset, TokenizedPairDataset]:
    training_records = build_tokenized_records(
        training_pairs,
        tokenizer,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
    )
    validation_records = build_tokenized_records(
        validation_pairs,
        tokenizer,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
    )
    return TokenizedPairDataset(training_records), TokenizedPairDataset(validation_records)


def _require_training_dependencies() -> dict[str, Any]:
    missing: list[str] = []
    modules: dict[str, Any] = {}

    for module_name in ("torch", "transformers", "peft"):
        try:
            modules[module_name] = import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        raise ImportError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install them on the DGX training environment before running this script."
        )

    return modules


def get_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("transformers", "peft", "torch"):
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            versions[module_name] = "not installed"
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def _build_training_arguments(
    training_arguments_cls: Any,
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Any:
    parameter_names = inspect.signature(training_arguments_cls.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 25,
        "save_total_limit": 1,
        "predict_with_generate": True,
        "remove_unused_columns": False,
        "seed": seed,
    }

    if "evaluation_strategy" in parameter_names:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in parameter_names:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in parameter_names:
        kwargs["logging_strategy"] = "steps"
    if "report_to" in parameter_names:
        kwargs["report_to"] = "none"

    try:
        return training_arguments_cls(**kwargs)
    except TypeError:
        if kwargs.get("report_to") == "none":
            kwargs["report_to"] = []
            return training_arguments_cls(**kwargs)
        raise


def create_lora_model(model_name: str) -> tuple[Any, Any]:
    modules = _require_training_dependencies()
    transformers = modules["transformers"]
    peft = modules["peft"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q", "v"],
        bias="none",
    )
    model = peft.get_peft_model(model, lora_config)
    return model, tokenizer


def _build_trainer_kwargs(
    trainer_cls: Any,
    *,
    model: Any,
    training_arguments: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
) -> dict[str, Any]:
    parameter_names = inspect.signature(trainer_cls.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    if "tokenizer" in parameter_names:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameter_names:
        kwargs["processing_class"] = tokenizer

    return kwargs


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch


def generate_cnl_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 64,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = getattr(model, "device", None)
    if device is not None:
        encoded = _move_batch_to_device(encoded, device)

    generated = model.generate(**encoded, max_new_tokens=max_new_tokens)
    return [
        text.strip()
        for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
    ]


def create_compiler() -> "CNLCompiler":
    from src.compiler.compiler import CNLCompiler

    return CNLCompiler()


def collect_sanity_check_rows(
    model: Any,
    tokenizer: Any,
    held_out_pairs: list[TrainingPair],
    *,
    compiler: "CNLCompiler | None" = None,
    sample_size: int = DEFAULT_SANITY_CHECK_SAMPLES,
) -> list[dict[str, str]]:
    if not held_out_pairs:
        return []

    compiler = compiler or create_compiler()
    sample = held_out_pairs[:sample_size]
    return collect_generation_rows(model, tokenizer, sample, compiler=compiler)


def collect_generation_rows(
    model: Any,
    tokenizer: Any,
    pairs: list[TrainingPair],
    *,
    compiler: "CNLCompiler | None" = None,
) -> list[dict[str, str]]:
    if not pairs:
        return []

    compiler = compiler or create_compiler()
    sample = list(pairs)
    prompts = [format_prompt(pair.nl) for pair in sample]
    generated_cnls = generate_cnl_outputs(model, tokenizer, prompts)

    rows: list[dict[str, str]] = []
    for pair, cnl in zip(sample, generated_cnls, strict=True):
        try:
            kif = compiler.compile(cnl)
        except Exception as exc:
            kif = f"<compile error: {exc}>"

        row = {"nl": pair.nl, "cnl": cnl, "kif": kif}
        if pair.pattern:
            row["pattern"] = pair.pattern
        rows.append(row)

    return rows


def build_pattern_coverage_pairs(all_pairs: list[TrainingPair]) -> tuple[list[TrainingPair], list[str]]:
    available_patterns = {pair.pattern for pair in all_pairs if pair.pattern}
    probes: list[TrainingPair] = []
    skipped_patterns: list[str] = []

    for pattern, nl in PATTERN_COVERAGE_SENTENCES:
        if pattern not in available_patterns:
            skipped_patterns.append(pattern)
            continue
        probes.append(TrainingPair(nl=nl, cnl="", pattern=pattern))

    return probes, skipped_patterns


def collect_pattern_coverage_rows(
    model: Any,
    tokenizer: Any,
    all_pairs: list[TrainingPair],
    *,
    compiler: "CNLCompiler | None" = None,
) -> tuple[list[dict[str, str]], list[str]]:
    probes, skipped_patterns = build_pattern_coverage_pairs(all_pairs)
    rows = collect_generation_rows(model, tokenizer, probes, compiler=compiler)
    return rows, skipped_patterns


def print_sanity_check(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No held-out samples available for sanity check.")
        return

    print("\nSanity check on held-out validation sentences:")
    for index, row in enumerate(rows, start=1):
        print(f"[{index}] NL:  {row['nl']}")
        print(f"    CNL: {row['cnl']}")
        print(f"    KIF: {row['kif']}")


def print_pattern_coverage_check(rows: list[dict[str, str]], skipped_patterns: list[str]) -> None:
    print("\nPattern coverage check:")
    if not rows:
        print("No pattern probes available.")
    else:
        for index, row in enumerate(rows, start=1):
            label = row.get("pattern", f"probe-{index}")
            print(f"[{label}] NL:  {row['nl']}")
            print(f"    CNL: {row['cnl']}")
            print(f"    KIF: {row['kif']}")

    if skipped_patterns:
        print(f"Skipped patterns: {', '.join(skipped_patterns)}")


def train(args: argparse.Namespace) -> None:
    modules = _require_training_dependencies()
    transformers = modules["transformers"]

    versions = get_dependency_versions()
    print("Dependency versions:")
    print(f"  transformers: {versions['transformers']}")
    print(f"  peft:         {versions['peft']}")
    print(f"  torch:        {versions['torch']}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_training_pairs(args.data_path, limit=args.limit)
    training_pairs, validation_pairs = split_training_pairs(pairs, seed=args.seed)

    print(f"\nLoaded pairs: {len(pairs):,}")
    print(f"Train split:  {len(training_pairs):,}")
    print(f"Val split:    {len(validation_pairs):,}")

    model, tokenizer = create_lora_model(args.model_name)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset, eval_dataset = build_datasets(
        training_pairs,
        validation_pairs,
        tokenizer,
        source_max_length=args.source_max_length,
        target_max_length=args.target_max_length,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    training_arguments = _build_training_arguments(
        transformers.Seq2SeqTrainingArguments,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    trainer = transformers.Seq2SeqTrainer(
        **_build_trainer_kwargs(
            transformers.Seq2SeqTrainer,
            model=model,
            training_arguments=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    compiler = create_compiler()

    rows = collect_sanity_check_rows(model, tokenizer, validation_pairs, compiler=compiler)
    print_sanity_check(rows)

    coverage_rows, skipped_patterns = collect_pattern_coverage_rows(
        model,
        tokenizer,
        pairs,
        compiler=compiler,
    )
    print_pattern_coverage_check(coverage_rows, skipped_patterns)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
