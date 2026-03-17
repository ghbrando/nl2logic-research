from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    description: str
    prepare_limit: int
    train_limit: int
    epochs: int
    batch_size: int
    learning_rate: float
    train_file: str
    output_dir: str
    session_name: str


_PRESETS: dict[str, TrainingPreset] = {
    "smoke": TrainingPreset(
        name="smoke",
        description="Fast smoke run over a small prepared artifact.",
        prepare_limit=7_000,
        train_limit=7_000,
        epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        train_file="data/training_pairs/train_balanced_7k.jsonl",
        output_dir="models/flan-t5-small-cnl-smoke",
        session_name="nl2logic-smoke",
    ),
    "base": TrainingPreset(
        name="base",
        description="Current default training run over the prepared 50k artifact.",
        prepare_limit=50_000,
        train_limit=50_000,
        epochs=3,
        batch_size=8,
        learning_rate=5e-4,
        train_file="data/training_pairs/train_balanced_50k.jsonl",
        output_dir="models/flan-t5-small-cnl",
        session_name="nl2logic-base",
    ),
    "full": TrainingPreset(
        name="full",
        description="Full filtered corpus run without the 50k cap.",
        prepare_limit=0,
        train_limit=0,
        epochs=3,
        batch_size=8,
        learning_rate=5e-4,
        train_file="data/training_pairs/train_full.jsonl",
        output_dir="models/flan-t5-small-cnl-full",
        session_name="nl2logic-full",
    ),
}


def list_training_presets() -> list[TrainingPreset]:
    return [_PRESETS[name] for name in ("smoke", "base", "full")]


def get_training_preset(name: str) -> TrainingPreset:
    normalized = name.strip().lower()
    try:
        return _PRESETS[normalized]
    except KeyError as exc:
        available = ", ".join(preset.name for preset in list_training_presets())
        raise ValueError(f"Unknown training preset {name!r}. Choose from: {available}") from exc