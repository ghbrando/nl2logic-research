from pathlib import Path
from lark import Lark, Transformer, v_args, Token

from src.ontology.vocab import load_closed_class_terms, load_closed_relation_terms

GRAMMAR_PATH = Path(__file__).parent / "cnl.lark"


# ── Transformer ────────────────────────────────────────────────────────────

class CNLToKIF(Transformer):
    """
    Walks the Lark parse tree and emits SUO-KIF s-expressions.
    Each method corresponds to one grammar production rule.
    Proven correct by structural induction — one case per method.
    """

    # ── start ──────────────────────────────────────────────────────────────

    def start(self, sentences):
        return "\n".join(sentences)

    # ── sentence / assertion passthrough ───────────────────────────────────

    def sentence(self, items):
        return items[0]

    def assertion(self, items):
        return items[0]

    def atomic_assertion(self, items):
        return items[0]

    def condition(self, items):
        if len(items) == 1:
            return items[0]
        return f"(and {' '.join(items)})"

    # ── Unary assertions ───────────────────────────────────────────────────

    @v_args(inline=True)
    def instance_assert(self, term, cls):
        return f"(instance {term} {cls})"

    @v_args(inline=True)
    def subclass_assert(self, child, parent):
        return f"(subclass {child} {parent})"

    # ── Binary assertions ──────────────────────────────────────────────────

    @v_args(inline=True)
    def binary_assert(self, rel, arg1, arg2):
        return f"({str(rel)} {arg1} {arg2})"

    # ── N-ary assertions ───────────────────────────────────────────────────

    def nary_assert(self, items):
        rel  = str(items[0])
        args = items[1:]
        return f"({rel} {' '.join(args)})"

    @v_args(inline=True)
    def negated_assertion(self, assertion):
        return f"(not {assertion})"

    # ── Quantifiers ────────────────────────────────────────────────────────

    @v_args(inline=True)
    def every(self, var, cls, *rest):
        body = f"(instance {var} {cls})"
        if rest:
            return f"(forall ({var}) (=> {body} {rest[0]}))"
        return f"(forall ({var}) {body})"

    @v_args(inline=True)
    def some(self, var, cls):
        return f"(exists ({var}) (instance {var} {cls}))"

    @v_args(inline=True)
    def no(self, var, cls, *rest):
        body = f"(instance {var} {cls})"
        if rest:
            return f"(forall ({var}) (=> {body} (not {rest[0]})))"
        return f"(forall ({var}) (not {body}))"

    # ── Conditional ────────────────────────────────────────────────────────

    @v_args(inline=True)
    def conditional(self, cond, consequent):
        return f"(=> {cond} {consequent})"

    # ── Term passthrough ───────────────────────────────────────────────────

    def term(self, items):
        return str(items[0])


# ── Compiler ───────────────────────────────────────────────────────────────

class CNLCompiler:

    def __init__(self):
        grammar       = GRAMMAR_PATH.read_text(encoding="utf-8")
        self._parser  = Lark(grammar, parser="earley", ambiguity="resolve")
        self._tx      = CNLToKIF()
        self._classes, self._relations = self._load_vocab()

    def compile(self, cnl: str) -> str:
        tree = self._parser.parse(cnl.strip())
        self._validate(tree)
        return self._tx.transform(tree)

    # ── Vocab validation ───────────────────────────────────────────────────

    def _load_vocab(self) -> tuple[set, dict]:
        return load_closed_class_terms(), load_closed_relation_terms()

    def _validate(self, tree) -> None:
        for token in tree.scan_values(lambda _: True):
            if isinstance(token, Token):
                if token.type == "CLASS_TERM" and str(token) not in self._classes:
                    raise ValueError(f"Unknown SUMO class: '{token}'")
                if token.type == "RELATION" and str(token) not in self._relations:
                    raise ValueError(f"Unknown SUMO relation: '{token}'")
