from __future__ import annotations

import logging
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.training.train import format_prompt, generate_cnl_outputs

if TYPE_CHECKING:
    from src.eval.evaluate import GoldPair

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "flan-t5-small-cnl"
_LOGGER = logging.getLogger(__name__)
_VALID_DECODING_MODES = {"auto", "constrained", "raw"}


def _require_inference_dependencies() -> tuple[Any, Any | None]:
    try:
        transformers = import_module("transformers")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "transformers is required for model-backed evaluation. "
            "Install it in the evaluation environment before using ModelPredictor."
        ) from exc

    try:
        peft = import_module("peft")
    except ModuleNotFoundError:
        peft = None

    return transformers, peft


def load_model_and_tokenizer(model_path: Path) -> tuple[Any, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_path}")

    transformers, peft = _require_inference_dependencies()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))

    model = None
    if peft is not None and hasattr(peft, "AutoPeftModelForSeq2SeqLM"):
        try:
            model = peft.AutoPeftModelForSeq2SeqLM.from_pretrained(str(model_path))
        except Exception:
            model = None

    if model is None:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    if hasattr(model, "eval"):
        model.eval()

    return model, tokenizer


class ModelPredictor:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_loader: Callable[[Path], tuple[Any, Any]] = load_model_and_tokenizer,
        max_new_tokens: int = 64,
        decoding: str = "auto",
        sampler_factory: Callable[[Any, Any], Any] = CNLSampler,
    ):
        self._model_path = Path(model_path)
        self._max_new_tokens = max_new_tokens
        self._decoding = decoding.lower()
        if self._decoding not in _VALID_DECODING_MODES:
            valid = ", ".join(sorted(_VALID_DECODING_MODES))
            raise ValueError(f"Unknown decoding mode {decoding!r}. Expected one of: {valid}")

        if model is None or tokenizer is None:
            model, tokenizer = model_loader(self._model_path)

        self._model = model
        self._tokenizer = tokenizer
        self._sampler_factory = sampler_factory
        self._sampler = None
        self._constrained_unavailable = False

    def _generate_raw(self, prompt: str) -> str | None:
        torch_module = None
        try:
            torch_module = import_module("torch")
        except ModuleNotFoundError:
            torch_module = None

        no_grad = torch_module.no_grad if torch_module is not None else nullcontext
        with no_grad():
            outputs = generate_cnl_outputs(
                self._model,
                self._tokenizer,
                [prompt],
                max_new_tokens=self._max_new_tokens,
            )
        return outputs[0] if outputs else None

    def _get_sampler(self):
        if self._sampler is None:
            self._sampler = self._sampler_factory(self._model, self._tokenizer)
        return self._sampler

    def _generate_constrained(self, prompt: str) -> str | None:
        sampler = self._get_sampler()
        return sampler.sample(prompt, max_tokens=self._max_new_tokens)

    def __call__(self, pair: "GoldPair") -> str | None:
        if CNLSampler.abstain_if_unsupported(pair.nl):
            return None

        prompt = format_prompt(pair.nl)
        if self._decoding != "raw" and not self._constrained_unavailable:
            try:
                return self._generate_constrained(prompt)
            except UnsupportedInputError:
                return None
            except ImportError as exc:
                if self._decoding == "constrained":
                    raise ImportError(
                        "Constrained decoding is unavailable in this environment. "
                        "Install outlines>=1.0 on Python <3.14, or rerun with '--decoding raw'."
                    ) from exc
                self._constrained_unavailable = True
                _LOGGER.warning(
                    "Constrained decoding is unavailable; falling back to raw generation for evaluation: %s",
                    exc,
                )

        return self._generate_raw(prompt)
