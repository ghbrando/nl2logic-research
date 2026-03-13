"""Evaluation utilities for NL2Logic."""

from .evaluate import (
    DEFAULT_GOLD_PATH,
    EvaluationReport,
    GoldPair,
    PatternMetrics,
    evaluate_pairs,
    load_gold_pairs,
    validate_gold_pairs,
)
from .model_predictor import DEFAULT_MODEL_PATH, ModelPredictor

__all__ = [
    "DEFAULT_GOLD_PATH",
    "DEFAULT_MODEL_PATH",
    "EvaluationReport",
    "GoldPair",
    "ModelPredictor",
    "PatternMetrics",
    "evaluate_pairs",
    "load_gold_pairs",
    "validate_gold_pairs",
]
