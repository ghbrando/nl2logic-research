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
   Wraps a HuggingFace seq2seq model with a custom token-prefix constraint
   automaton for the canonical CNL fragment used in training/evaluation.
   validate_output() always delegates to CNLCompiler.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib import import_module
import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

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
_CONTROL_LITERAL_GRAMMAR_RE = re.compile(r'^\s*root\s*::=\s*(?P<literal>".*")\s*$', re.DOTALL)
_CANONICAL_VAR_TERMS = ("?x", "?y", "?z", "?a", "?b", "?c")
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


@dataclass
class _ConstraintNode:
    token_edges: dict[int, set[int]] = field(default_factory=dict)
    epsilon: set[int] = field(default_factory=set)
    accept: bool = False


class _ConstraintNFA:
    def __init__(self):
        self.nodes: list[_ConstraintNode] = []

    def new_node(self) -> int:
        self.nodes.append(_ConstraintNode())
        return len(self.nodes) - 1

    def add_epsilon(self, src: int, dst: int) -> None:
        self.nodes[src].epsilon.add(dst)

    def add_token_edge(self, src: int, token_id: int, dst: int) -> None:
        self.nodes[src].token_edges.setdefault(token_id, set()).add(dst)


class _ConstraintBuilder:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._nfa = _ConstraintNFA()
        self._segment_cache: dict[tuple[tuple[int, ...], ...], tuple[int, frozenset[int]]] = {}
        self.start_state = self._nfa.new_node()
        self.accept_state = self._nfa.new_node()
        self._nfa.nodes[self.accept_state].accept = True

    @property
    def nfa(self) -> _ConstraintNFA:
        return self._nfa

    def _encode_surface(self, text: str) -> tuple[int, ...]:
        encoded = self._tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return tuple(int(token_id) for token_id in input_ids)

    def _build_segment_graph(
        self,
        sequences: Iterable[tuple[int, ...]],
    ) -> tuple[int, frozenset[int]]:
        unique_sequences = tuple(sorted(set(sequences)))
        if unique_sequences in self._segment_cache:
            return self._segment_cache[unique_sequences]

        root_state = self._nfa.new_node()
        terminal_states: set[int] = set()

        for sequence in unique_sequences:
            current_state = root_state
            if not sequence:
                terminal_states.add(current_state)
                continue

            for token_id in sequence:
                next_states = self._nfa.nodes[current_state].token_edges.get(token_id)
                if next_states and len(next_states) == 1:
                    next_state = next(iter(next_states))
                else:
                    next_state = self._nfa.new_node()
                    self._nfa.add_token_edge(current_state, token_id, next_state)
                current_state = next_state
            terminal_states.add(current_state)

        cached = (root_state, frozenset(terminal_states))
        self._segment_cache[unique_sequences] = cached
        return cached

    def add_template(self, segment_groups: list[Iterable[tuple[int, ...]]]) -> None:
        current_state = self.start_state
        for index, sequences in enumerate(segment_groups):
            next_state = self.accept_state if index == len(segment_groups) - 1 else self._nfa.new_node()
            segment_root, segment_terminals = self._build_segment_graph(sequences)
            self._nfa.add_epsilon(current_state, segment_root)
            for terminal_state in segment_terminals:
                self._nfa.add_epsilon(terminal_state, next_state)
            current_state = next_state


class _PrefixConstraint:
    def __init__(
        self,
        nfa: _ConstraintNFA,
        *,
        start_state: int,
        eos_token_id: int | None,
        decoder_start_token_id: int | None,
    ):
        self._nfa = nfa
        self._start_state = start_state
        self._eos_token_id = eos_token_id
        self._decoder_start_token_id = decoder_start_token_id
        self._closure_cache: dict[tuple[int, ...], frozenset[int]] = {}
        self._state_cache: dict[tuple[int, ...], frozenset[int]] = {
            tuple(): self._epsilon_closure((self._start_state,))
        }

    def _epsilon_closure(self, states: Iterable[int]) -> frozenset[int]:
        key = tuple(sorted(set(states)))
        if key in self._closure_cache:
            return self._closure_cache[key]

        seen = set(key)
        stack = list(key)
        while stack:
            state = stack.pop()
            for dst in self._nfa.nodes[state].epsilon:
                if dst not in seen:
                    seen.add(dst)
                    stack.append(dst)

        closure = frozenset(seen)
        self._closure_cache[key] = closure
        return closure

    def _advance(self, states: frozenset[int], token_id: int) -> frozenset[int]:
        destinations: set[int] = set()
        for state in states:
            destinations.update(self._nfa.nodes[state].token_edges.get(token_id, set()))
        return self._epsilon_closure(destinations)

    def _normalise_prefix(self, input_ids) -> tuple[int, ...]:
        if hasattr(input_ids, "tolist"):
            values = input_ids.tolist()
        else:
            values = list(input_ids)
        if values and isinstance(values[0], list):
            values = values[0]
        prefix = tuple(int(token_id) for token_id in values)
        if prefix and self._decoder_start_token_id is not None and prefix[0] == self._decoder_start_token_id:
            prefix = prefix[1:]
        return prefix

    def _states_for_prefix(self, prefix: tuple[int, ...]) -> frozenset[int]:
        if prefix in self._state_cache:
            return self._state_cache[prefix]

        previous = prefix[:-1]
        previous_states = self._states_for_prefix(previous)
        states = self._advance(previous_states, prefix[-1])
        self._state_cache[prefix] = states
        return states

    def __call__(self, _batch_id: int, input_ids) -> list[int]:
        prefix = self._normalise_prefix(input_ids)
        states = self._states_for_prefix(prefix)
        if not states:
            raise RuntimeError("No valid constrained continuation remains for the generated prefix.")

        allowed: set[int] = set()
        accepting = False
        for state in states:
            node = self._nfa.nodes[state]
            allowed.update(node.token_edges.keys())
            accepting = accepting or node.accept

        if accepting and self._eos_token_id is not None:
            allowed.add(int(self._eos_token_id))

        if not allowed:
            raise RuntimeError("Constraint automaton has no valid next tokens.")

        return sorted(allowed)


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
    Uses a Transformers-native prefix constraint function; no third-party
    constrained-decoding backend is required at sample time.
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
        self._prefix_constraint = None
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

    def _encode_surface(self, text: str) -> tuple[int, ...]:
        encoded = self._tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return tuple(int(token_id) for token_id in input_ids)

    def _fixed_segment(self, text: str) -> list[tuple[int, ...]]:
        return [self._encode_surface(text)]

    def _slot_sequences(self, terms: Iterable[str], *, leading_space: bool) -> list[tuple[int, ...]]:
        prefix = " " if leading_space else ""
        return [self._encode_surface(f"{prefix}{term}") for term in _sorted_terms(list(terms))]

    def _control_literal(self) -> str | None:
        match = _CONTROL_LITERAL_GRAMMAR_RE.match(self._grammar)
        if match is None:
            return None
        return json.loads(match.group("literal"))

    def _build_prefix_constraint(self) -> _PrefixConstraint:
        builder = _ConstraintBuilder(self._tokenizer)

        control_literal = self._control_literal()
        if control_literal is not None:
            builder.add_template([self._fixed_segment(control_literal)])
        else:
            classes = _load_terms(_SUMO_CLASSES)
            relations = _load_terms(_SUMO_RELATIONS)

            class_start = self._slot_sequences(classes, leading_space=False)
            class_space = self._slot_sequences(classes, leading_space=True)
            relation_start = self._slot_sequences(relations, leading_space=False)
            relation_space = self._slot_sequences(relations, leading_space=True)
            var_space = self._slot_sequences(_CANONICAL_VAR_TERMS, leading_space=True)
            term_space = class_space + var_space

            builder.add_template([self._fixed_segment("?x is-a"), class_space])
            builder.add_template([class_start, self._fixed_segment(" subclass-of"), class_space])
            builder.add_template([relation_start, term_space, term_space])
            builder.add_template(
                [
                    relation_start,
                    self._fixed_segment(" ["),
                    term_space,
                    self._fixed_segment(" ,"),
                    term_space,
                    self._fixed_segment(" ,"),
                    term_space,
                    self._fixed_segment(" ]"),
                ]
            )
            builder.add_template(
                [
                    self._fixed_segment("every ?x is-a"),
                    class_space,
                    self._fixed_segment(" implies"),
                    relation_space,
                    self._fixed_segment(" ?x"),
                    term_space,
                ]
            )
            builder.add_template(
                [
                    self._fixed_segment("if"),
                    relation_space,
                    self._fixed_segment(" ?x"),
                    term_space,
                    self._fixed_segment(" then ?x is-a"),
                    class_space,
                ]
            )
            builder.add_template([self._fixed_segment("some ?x is-a"), class_space])
            builder.add_template([self._fixed_segment("not ?x is-a"), class_space])
            builder.add_template([self._fixed_segment("not"), relation_space, term_space, term_space])
            builder.add_template(
                [
                    self._fixed_segment("not"),
                    relation_space,
                    self._fixed_segment(" ["),
                    term_space,
                    self._fixed_segment(" ,"),
                    term_space,
                    self._fixed_segment(" ,"),
                    term_space,
                    self._fixed_segment(" ]"),
                ]
            )

        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        decoder_start_token_id = getattr(getattr(self._hf_model, "config", None), "decoder_start_token_id", None)
        return _PrefixConstraint(
            builder.nfa,
            start_state=builder.start_state,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

    def _get_prefix_constraint(self) -> _PrefixConstraint:
        if self._prefix_constraint is None:
            self._prefix_constraint = self._build_prefix_constraint()
        return self._prefix_constraint

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
        prefix_constraint = self._get_prefix_constraint()

        try:
            torch_module = import_module("torch")
        except ModuleNotFoundError:
            torch_module = None

        no_grad = torch_module.no_grad if torch_module is not None else nullcontext
        with no_grad():
            generated = self._hf_model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                prefix_allowed_tokens_fn=prefix_constraint,
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
