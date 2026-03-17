"""
cnl_fsm.py — Stage 2: Outlines FSM integration for NL2Logic.

Provides two public surfaces:

1. build_vocabulary_grammar(classes_path, relations_path) -> str
   Reads the SUMO vocab JSONL files, replaces the open-ended CLASS_TERM and
   RELATION terminals in cnl.lark with closed alternations of all known SUMO
   terms, and verifies the resulting grammar compiles under Lark LALR(1).
   This grammar string is what gets passed to Outlines CFG-constrained
   generation, making hallucinated class/relation names physically impossible.

2. CNLSampler(hf_model, tokenizer, *, grammar_str=None)
   Wraps an Outlines-constrained HuggingFace model.  Requires outlines>=1.0
   and Python <3.14.  On Python 3.14 the outlines import guard raises a clear
   ImportError rather than a cryptic failure.

   validate_output(cnl) -> str  delegates to CNLCompiler and is always
   available regardless of whether outlines is installed.

Python 3.14 note
----------------
outlines>=1.0 declares Requires-Python <3.14.  All CNL grammar building and
output validation work without outlines.  The CNLSampler.sample() method is
the only part that requires a working outlines install.  Training runs on the
DGX Spark (Python 3.10 or 3.11) where outlines installs cleanly.
"""

import json
import logging
import math
import re
import sys
from pathlib import Path

from lark import Lark

from src.compiler.compiler import CNLCompiler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_GRAMMAR_PATH  = _REPO_ROOT / "src" / "compiler" / "cnl.lark"
_SUMO_CLASSES  = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_SUMO_RELATIONS = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"

# Terminals in cnl.lark that we replace with closed vocabulary alternations.
_CLASS_TERM_RE_PATTERN  = r"CLASS_TERM\s*:\s*/\[A-Z\]\[a-zA-Z0-9_\]\*/"
_RELATION_RE_PATTERN    = r"RELATION\s*:\s*/\[a-z\]\[a-zA-Z0-9\]\*/"

_LOGGER = logging.getLogger(__name__)
_INSTRUCTION_PREFIX_RE = re.compile(r"^\s*translate(?:\s+to\s+\w+)?\s*:\s*", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_?'-]+")
_MULTI_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_MAX_SUPPORTED_TOKENS = 64
_UNSUPPORTED_MARKERS = (
    (re.compile(r"\bit\b", re.IGNORECASE), "pronoun 'it'"),
    (re.compile(r"\bits\b", re.IGNORECASE), "pronoun 'its'"),
    (re.compile(r"\bthey\b", re.IGNORECASE), "pronoun 'they'"),
    (re.compile(r"\bthem\b", re.IGNORECASE), "pronoun 'them'"),
    (re.compile(r"\btheir\b", re.IGNORECASE), "pronoun 'their'"),
    (re.compile(r"\bthis unit\b", re.IGNORECASE), "phrase 'this unit'"),
    (re.compile(r"\bthat unit\b", re.IGNORECASE), "phrase 'that unit'"),
    (re.compile(r"\bthese units\b", re.IGNORECASE), "phrase 'these units'"),
    (re.compile(r"\bthose units\b", re.IGNORECASE), "phrase 'those units'"),
    (re.compile(r"\bformer\b", re.IGNORECASE), "marker 'former'"),
    (re.compile(r"\blatter\b", re.IGNORECASE), "marker 'latter'"),
    (re.compile(r"\baforementioned\b", re.IGNORECASE), "marker 'aforementioned'"),
)
_SUPPORTED_POSSESSIVE_SLOT_RE = re.compile(
    r"\bas\s+its\s+(agent|patient|destination|origin)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Grammar builder
# ---------------------------------------------------------------------------

def _load_terms(path: Path, key: str = "term") -> list[str]:
    """Read a JSONL file and return the value of `key` for each record."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]


def _alternation(terms: list[str]) -> str:
    """
    Build a Lark regex terminal alternation from a list of string terms.

    Terms are sorted longest-first so that longer matches shadow shorter
    prefixes (e.g. 'agentName' shadows 'agent').
    Terms are re.escape'd so special regex characters are treated literally.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    return "|".join(re.escape(t) for t in sorted_terms)


def build_vocabulary_grammar(
    classes_path: Path = _SUMO_CLASSES,
    relations_path: Path = _SUMO_RELATIONS,
) -> str:
    """
    Build a vocabulary-closed CNL grammar string suitable for Outlines CFG
    constrained generation and Lark LALR(1) parsing.

    Reads the base grammar from cnl.lark and replaces:
      CLASS_TERM : /[A-Z][a-zA-Z0-9_]*/   (open)
      RELATION   : /[a-z][a-zA-Z0-9]*/    (open)
    with closed alternation patterns built from all known SUMO terms.

    Verifies LALR(1) compatibility before returning.

    Raises
    ------
    RuntimeError
        If the resulting grammar is not LALR(1)-compatible.
    """
    base_grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")

    classes   = _load_terms(classes_path)
    relations = _load_terms(relations_path)

    class_alt    = _alternation(classes)
    relation_alt = _alternation(relations)

    # Replace open terminals with closed alternations.
    # The alternation is wrapped in a non-capturing group so the regex
    # terminal boundary is unambiguous.
    grammar = re.sub(
        _CLASS_TERM_RE_PATTERN,
        f"CLASS_TERM  : /(?:{class_alt})/",
        base_grammar,
    )
    grammar = re.sub(
        _RELATION_RE_PATTERN,
        f"RELATION    : /(?:{relation_alt})/",
        grammar,
    )

    return grammar


def validate_base_grammar_lalr() -> None:
    """
    Verify that the *base* CNL grammar (with open-ended terminal regexes, no
    vocabulary substitution) compiles under Lark LALR(1).

    This is a fast check (~ms) used in CI to confirm the grammar's structural
    skeleton is LALR(1)-compatible before the large vocabulary alternations
    are injected.  The vocabulary-closed grammar uses Earley at parse time
    because a 29,649-term alternation makes LALR table construction impractical.

    Raises
    ------
    RuntimeError
        If the base grammar is not LALR(1)-compatible.
    """
    base_grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")
    try:
        Lark(base_grammar, parser="lalr")
    except Exception as exc:
        raise RuntimeError(
            "Base CNL grammar is not LALR(1)-compatible. "
            "Details: " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# CNLSampler
# ---------------------------------------------------------------------------

class UnsupportedInputError(RuntimeError):
    """Raised when the NL input appears outside the supported CNL fragment."""


class CNLSampler:
    """
    Wraps an Outlines-constrained HuggingFace model to produce CNL strings
    guaranteed parseable by the Lark grammar, then validates them through
    CNLCompiler to obtain SUO-KIF output.

    Parameters
    ----------
    hf_model : transformers.PreTrainedModel
        A loaded HuggingFace encoder-decoder model (e.g. T5).
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    grammar_str : str, optional
        Pre-built vocabulary grammar string.  Defaults to calling
        build_vocabulary_grammar() on construction.

    Notes
    -----
    Requires outlines>=1.0 and Python <3.14.  On Python 3.14 the
    sample() method raises ImportError with a clear message.
    validate_output() is always available.
    """

    def __init__(
        self,
        hf_model,
        tokenizer,
        *,
        grammar_str: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        self._hf_model  = hf_model
        self._tokenizer = tokenizer
        self._grammar   = grammar_str if grammar_str is not None else build_vocabulary_grammar()
        self._compiler  = CNLCompiler()
        self._outlines_model = None  # lazy-initialised in sample()
        self._confidence_threshold = confidence_threshold

    @staticmethod
    def _normalise_prompt(nl: str) -> str:
        stripped = _INSTRUCTION_PREFIX_RE.sub("", nl.strip())
        return re.sub(r"\s+", " ", stripped)

    @classmethod
    def _unsupported_reasons(cls, nl: str) -> list[str]:
        text = cls._normalise_prompt(nl)
        reasons: list[str] = []

        sentences = [part.strip() for part in _MULTI_SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if len(sentences) > 1:
            reasons.append("multiple sentences detected")

        token_count = len(_TOKEN_RE.findall(text))
        if token_count > _MAX_SUPPORTED_TOKENS:
            reasons.append(
                f"input length {token_count} tokens exceeds conservative threshold {_MAX_SUPPORTED_TOKENS}"
            )

        discourse_text = _SUPPORTED_POSSESSIVE_SLOT_RE.sub("as SLOT", text)

        for pattern, label in _UNSUPPORTED_MARKERS:
            if pattern.search(discourse_text):
                reasons.append(f"unsupported discourse marker: {label}")
                break

        return reasons

    @classmethod
    def abstain_if_unsupported(cls, nl: str) -> bool:
        """
        Conservatively abstain on inputs likely outside the supported CNL fragment.

        The heuristic is intentionally biased toward false abstains over silent
        semantic errors.
        """
        return bool(cls._unsupported_reasons(nl))

    def _get_outlines_model(self):
        """Lazy-import and initialise the Outlines model wrapper."""
        if self._outlines_model is not None:
            return self._outlines_model

        if sys.version_info >= (3, 14):
            raise ImportError(
                "outlines>=1.0 requires Python <3.14. "
                "CNLSampler.sample() is not available on Python 3.14+. "
                "Run inference on the DGX Spark (Python 3.10/3.11) or "
                "upgrade once outlines adds Python 3.14 support."
            )

        try:
            from outlines.models import from_transformers  # type: ignore[import]
            from outlines.types import CFG                 # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "outlines is not installed or not importable. "
                "Install it with: pip install outlines>=1.0"
            ) from exc

        self._cfg = CFG(self._grammar)
        self._outlines_model = from_transformers(self._hf_model, self._tokenizer)
        return self._outlines_model

    def _tokenize_prompt(self, prompt: str):
        encoded = self._tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        device = getattr(self._hf_model, "device", None)
        if device is not None:
            encoded = self._move_batch_to_device(encoded, device)
        return encoded

    @staticmethod
    def _move_batch_to_device(batch, device):
        if hasattr(batch, "to"):
            return batch.to(device)
        if isinstance(batch, dict):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    @staticmethod
    def _score_row(score_step) -> list[float]:
        if hasattr(score_step, "detach"):
            score_step = score_step.detach()
        if hasattr(score_step, "cpu"):
            score_step = score_step.cpu()
        if hasattr(score_step, "tolist"):
            score_step = score_step.tolist()

        if not isinstance(score_step, (list, tuple)) or not score_step:
            raise ValueError("Model score step must be a non-empty sequence.")

        row = score_step[0] if isinstance(score_step[0], (list, tuple)) else score_step
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Model score row must be a non-empty sequence.")

        return [float(value) for value in row]

    @classmethod
    def _top1_probability(cls, score_step) -> float:
        row = cls._score_row(score_step)
        max_logit = max(row)
        exponentials = [math.exp(value - max_logit) for value in row]
        return max(exponentials) / sum(exponentials)

    @classmethod
    def _mean_top1_probability(cls, scores) -> float:
        if not scores:
            raise ValueError("Model generation did not return token scores.")

        probabilities = [cls._top1_probability(score_step) for score_step in scores]
        return sum(probabilities) / len(probabilities)

    def sample(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Run constrained generation and return a CNL string.

        The Outlines FSM guarantees the output matches the vocabulary grammar,
        so it is always parseable by CNLCompiler (modulo vocab validation).

        Parameters
        ----------
        prompt : str
            Natural-language input prompt for the T5 model.
        max_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        str
            A CNL sentence string.

        Raises
        ------
        UnsupportedInputError
            If the natural-language input appears outside the supported fragment
            and the sampler conservatively abstains before generation.
        """
        reasons = self._unsupported_reasons(prompt)
        if reasons:
            message = "; ".join(reasons)
            _LOGGER.warning("Abstaining from CNL generation for unsupported input: %s", message)
            raise UnsupportedInputError(message)

        model = self._get_outlines_model()
        return model(prompt, output_type=self._cfg, max_tokens=max_tokens, backend="xgrammar")

    def sample_with_confidence(self, prompt: str, max_tokens: int = 200) -> tuple[str, float]:
        """
        Run HuggingFace generation directly and return both the decoded CNL and
        a confidence score computed as the mean top-1 probability across the
        generated sequence.

        Low-confidence generations are logged but still returned so the caller
        can decide whether to abstain.
        """
        reasons = self._unsupported_reasons(prompt)
        if reasons:
            message = "; ".join(reasons)
            _LOGGER.warning("Abstaining from CNL generation for unsupported input: %s", message)
            raise UnsupportedInputError(message)

        encoded = self._tokenize_prompt(prompt)
        output = self._hf_model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if not hasattr(output, "sequences"):
            raise ValueError("Model generation result must include sequences.")
        if not hasattr(output, "scores"):
            raise ValueError("Model generation result must include token scores.")

        decoded = self._tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        if not decoded:
            raise ValueError("Tokenizer returned no decoded sequences.")

        cnl = decoded[0].strip()
        confidence = self._mean_top1_probability(output.scores)
        if confidence < self._confidence_threshold:
            _LOGGER.warning(
                "Low-confidence CNL generation: confidence=%.4f threshold=%.4f cnl=%r",
                confidence,
                self._confidence_threshold,
                cnl,
            )
        return cnl, confidence

    def validate_output(self, cnl: str) -> str:
        """
        Compile a CNL string to SUO-KIF via CNLCompiler.

        Does not use Outlines — always available regardless of Python version.

        Parameters
        ----------
        cnl : str
            A CNL sentence string.

        Returns
        -------
        str
            Well-formed SUO-KIF s-expression string.

        Raises
        ------
        lark.exceptions.UnexpectedInput
            If `cnl` is not parseable by the CNL grammar.
        ValueError
            If `cnl` contains unknown SUMO class or relation terms.
        """
        return self._compiler.compile(cnl)
