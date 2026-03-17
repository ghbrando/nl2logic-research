"""
cnl_fsm.py - constrained decoding integration for NL2Logic.

This module keeps two grammar builders because the parser/compiler path and
the constrained-decoding backend accept different syntaxes:

1. build_vocabulary_grammar(...)
   Produces the existing Lark grammar with CLASS_TERM and RELATION closed over
   the SUMO vocabulary. This remains the source of truth for parsing and
   compiler-side validation.

2. build_xgrammar_grammar(...)
   Produces an xgrammar-compatible GBNF grammar that emits the same CNL
   fragment in a canonical one-line surface form. This is what constrained
   decoding uses, because xgrammar follows GBNF rather than Lark syntax.

3. CNLSampler(...)
   Wraps a HuggingFace seq2seq model with xgrammar-backed logits masking.
   validate_output() always delegates to CNLCompiler regardless of whether
   xgrammar is installed.
"""

from contextlib import nullcontext
from importlib import import_module
import json
import logging
import math
import re
from pathlib import Path

from lark import Lark

from src.compiler.compiler import CNLCompiler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAMMAR_PATH = _REPO_ROOT / "src" / "compiler" / "cnl.lark"
_SUMO_CLASSES = _REPO_ROOT / "data" / "training_pairs" / "sumo_classes.jsonl"
_SUMO_RELATIONS = _REPO_ROOT / "data" / "training_pairs" / "sumo_relations.jsonl"

# Terminals in cnl.lark that we replace with closed vocabulary alternations.
_CLASS_TERM_RE_PATTERN = r"CLASS_TERM\s*:\s*/\[A-Z\]\[a-zA-Z0-9_\]\*/"
_RELATION_RE_PATTERN = r"RELATION\s*:\s*/\[a-z\]\[a-zA-Z0-9\]\*/"

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
_XGRAMMAR_TEMPLATE = """
root ::= sentence
sentence ::= assertion | quantified | conditional
assertion ::= atomic-assertion | "not " atomic-assertion
atomic-assertion ::= unary-assert | nary-assert | binary-assert
unary-assert ::= term " is-a " class-term | class-term " subclass-of " class-term
binary-assert ::= relation " " term " " term
nary-assert ::= relation " [ " term (" , " term)+ " ]"
quantified ::= "every " var " is-a " class-term (" implies " sentence)?
             | "some " var " is-a " class-term
             | "no " var " is-a " class-term (" implies " sentence)?
conditional ::= "if " condition " then " sentence
condition ::= assertion (" and " assertion)*
term ::= var | class-term
var ::= "?" [a-z] [a-zA-Z0-9_]*
class-term ::= __CLASS_ALTS__
relation ::= __RELATION_ALTS__
""".strip()


# ---------------------------------------------------------------------------
# Grammar builders
# ---------------------------------------------------------------------------

def _load_terms(path: Path, key: str = "term") -> list[str]:
    """Read a JSONL file and return the value of `key` for each record."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]


def _sorted_terms(terms: list[str]) -> list[str]:
    """Sort terms longest-first, then lexicographically for stable output."""
    return sorted(set(terms), key=lambda term: (-len(term), term))


def _lark_regex_alternation(terms: list[str]) -> str:
    """
    Build a Lark regex terminal alternation from a list of string terms.

    Terms are sorted longest-first so longer matches shadow shorter prefixes
    (for example, 'agentName' before 'agent').
    """
    return "|".join(re.escape(term) for term in _sorted_terms(terms))


def _quote_gbnf_literal(term: str) -> str:
    """Quote a literal for GBNF / xgrammar using JSON escaping semantics."""
    return json.dumps(term)


def _gbnf_literal_alternation(terms: list[str]) -> str:
    """Build a GBNF alternation over exact quoted literals."""
    return " | ".join(_quote_gbnf_literal(term) for term in _sorted_terms(terms))


def build_vocabulary_grammar(
    classes_path: Path = _SUMO_CLASSES,
    relations_path: Path = _SUMO_RELATIONS,
) -> str:
    """
    Build the vocabulary-closed Lark grammar used by the compiler/parser path.

    Reads the base grammar from cnl.lark and replaces:
      CLASS_TERM : /[A-Z][a-zA-Z0-9_]*/   (open)
      RELATION   : /[a-z][a-zA-Z0-9]*/    (open)
    with closed alternation patterns built from all known SUMO terms.
    """
    base_grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")

    classes = _load_terms(classes_path)
    relations = _load_terms(relations_path)

    class_alt = _lark_regex_alternation(classes)
    relation_alt = _lark_regex_alternation(relations)

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


def build_xgrammar_grammar(
    classes_path: Path = _SUMO_CLASSES,
    relations_path: Path = _SUMO_RELATIONS,
) -> str:
    """
    Build a closed-vocabulary GBNF grammar for xgrammar-backed generation.

    The constrained grammar intentionally emits a single canonical CNL sentence
    with exact spacing. That keeps generation backend-compatible and aligns
    better with exact-match evaluation.
    """
    classes = _load_terms(classes_path)
    relations = _load_terms(relations_path)

    return (
        _XGRAMMAR_TEMPLATE
        .replace("__CLASS_ALTS__", _gbnf_literal_alternation(classes))
        .replace("__RELATION_ALTS__", _gbnf_literal_alternation(relations))
    )


def validate_base_grammar_lalr() -> None:
    """
    Verify that the base CNL grammar (with open terminals) compiles as LALR(1).

    This is a fast structural check used in CI before the large vocabulary
    alternations are injected.
    """
    base_grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")
    try:
        Lark(base_grammar, parser="lalr")
    except Exception as exc:
        raise RuntimeError(
            "Base CNL grammar is not LALR(1)-compatible. Details: " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# CNLSampler
# ---------------------------------------------------------------------------

class UnsupportedInputError(RuntimeError):
    """Raised when the NL input appears outside the supported CNL fragment."""


class CNLSampler:
    """
    Wrap a HuggingFace seq2seq model to produce canonical CNL.

    Parameters
    ----------
    hf_model : transformers.PreTrainedModel
        A loaded HuggingFace encoder-decoder model (for example, T5).
    tokenizer : transformers.PreTrainedTokenizer
        The corresponding tokenizer.
    grammar_str : str, optional
        Pre-built constrained grammar string. Defaults to build_xgrammar_grammar().

    Notes
    -----
    Requires xgrammar at sample time. validate_output() is always available.
    """

    def __init__(
        self,
        hf_model,
        tokenizer,
        *,
        grammar_str: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        self._hf_model = hf_model
        self._tokenizer = tokenizer
        self._grammar = grammar_str if grammar_str is not None else build_xgrammar_grammar()
        self._compiler = CNLCompiler()
        self._xgrammar_logits_processors = None
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
        Conservatively abstain on inputs likely outside the supported fragment.

        The heuristic is intentionally biased toward false abstains over silent
        semantic errors.
        """
        return bool(cls._unsupported_reasons(nl))

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

    def _resolve_vocab_size(self) -> int:
        config = getattr(self._hf_model, "config", None)
        model_vocab_size = getattr(config, "vocab_size", None)
        if isinstance(model_vocab_size, int) and model_vocab_size > 0:
            return model_vocab_size

        tokenizer_vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(tokenizer_vocab_size, int) and tokenizer_vocab_size > 0:
            return tokenizer_vocab_size

        if hasattr(self._tokenizer, "__len__"):
            derived_vocab_size = int(len(self._tokenizer))
            if derived_vocab_size > 0:
                return derived_vocab_size

        raise ValueError("Unable to determine tokenizer vocabulary size for constrained decoding.")

    def _get_xgrammar_logits_processors(self):
        """Lazy-import and initialise xgrammar-backed Hugging Face logits processors."""
        if self._xgrammar_logits_processors is not None:
            return self._xgrammar_logits_processors

        try:
            xgrammar = import_module("xgrammar")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "xgrammar is not installed or not importable. "
                "Install it with: pip install xgrammar"
            ) from exc

        try:
            hf_integration = import_module("xgrammar.contrib.hf")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "xgrammar's Hugging Face integration is unavailable. "
                "Install a recent xgrammar build that provides xgrammar.contrib.hf."
            ) from exc

        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
            self._tokenizer,
            vocab_size=self._resolve_vocab_size(),
        )
        grammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_grammar(self._grammar)
        processor = hf_integration.LogitsProcessor(compiled_grammar)

        try:
            transformers = import_module("transformers")
        except ModuleNotFoundError:
            processors = [processor]
        else:
            logits_processor_list_cls = getattr(transformers, "LogitsProcessorList", None)
            processors = (
                logits_processor_list_cls([processor])
                if logits_processor_list_cls is not None
                else [processor]
            )

        self._xgrammar_logits_processors = processors
        return self._xgrammar_logits_processors

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
        Run constrained generation and return a canonical CNL string.

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

        encoded = self._tokenize_prompt(prompt)
        logits_processors = self._get_xgrammar_logits_processors()

        try:
            torch_module = import_module("torch")
        except ModuleNotFoundError:
            torch_module = None

        no_grad = torch_module.no_grad if torch_module is not None else nullcontext
        with no_grad():
            generated = self._hf_model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                logits_processor=logits_processors,
            )

        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        if not decoded:
            raise ValueError("Tokenizer returned no decoded sequences.")
        return decoded[0].strip()

    def sample_with_confidence(self, prompt: str, max_tokens: int = 200) -> tuple[str, float]:
        """
        Run HuggingFace generation directly and return CNL plus a confidence score.

        The confidence is computed as the mean top-1 probability across the
        generated sequence.
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

        Does not use xgrammar and is always available regardless of
        constrained-decoding backend availability.
        """
        return self._compiler.compile(cnl)
