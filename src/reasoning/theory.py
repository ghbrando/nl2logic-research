"""Translate ground classification KIF to a deliberately small FOF theory.

This is not a general SUMO translator. Unsupported formulas fail closed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class UnsupportedFormula(ValueError):
    pass


@dataclass(frozen=True)
class Literal:
    relation: str
    left: str
    right: str
    negative: bool = False

    def negate(self) -> Literal:
        return Literal(self.relation, self.left, self.right, not self.negative)

    def tptp(self) -> str:
        # Quoted atoms preserve case. Strict input validation excludes escapes.
        atom = f"{self.relation}('{self.left}','{self.right}')"
        return f"~({atom})" if self.negative else atom


_ATOM = r"\(\s*(subclass|instance)\s+([A-Z][A-Za-z0-9_]*)\s+([A-Z][A-Za-z0-9_]*)\s*\)"


def parse_literal(kif: str) -> Literal:
    if not isinstance(kif, str):
        raise UnsupportedFormula("KIF must be a string")
    match = re.fullmatch(rf"\s*{_ATOM}\s*", kif)
    if match:
        return Literal(*match.groups())
    match = re.fullmatch(rf"\s*\(\s*not\s+{_ATOM}\s*\)\s*", kif)
    if match:
        return Literal(*match.groups(), negative=True)
    raise UnsupportedFormula(
        "Supported KIF: ground (subclass A B), (instance A B), or their explicit negation"
    )


RULES = {
    "rule_subclass_transitivity": {
        "formula": "![X,Y,Z]: ((subclass(X,Y) & subclass(Y,Z)) => subclass(X,Z))",
        "description": "Subclass membership is transitive.",
    },
    "rule_instance_inheritance": {
        "formula": "![X,Y,Z]: ((instance(X,Y) & subclass(Y,Z)) => instance(X,Z))",
        "description": "Instances inherit membership in superclasses.",
    },
}


def load_records(path: Path, kind: str) -> list[dict]:
    if kind not in {"source", "background"}:
        raise ValueError("Unknown evidence kind")
    records = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"{path}:{number}: missing or duplicate record_id")
        if row.get("status") not in {None, "reviewed", "accepted"}:
            raise ValueError(f"{path}:{number}: draft or unreviewed record")
        if row.get("formalizable") is False:
            raise ValueError(f"{path}:{number}: non-formalizable record")
        parse_literal(row.get("kif"))
        if kind == "source" and not (row.get("original") or row.get("nl")):
            raise ValueError(f"{path}:{number}: source passage is required")
        if kind == "background" and not row.get("rationale"):
            raise ValueError(f"{path}:{number}: background rationale is required")
        seen.add(record_id)
        records.append({**row, "kind": kind, "input_file": str(path.resolve()), "input_line": number})
    return records


def build_theory(records: list[dict]) -> tuple[str, dict[str, dict], set[str]]:
    lines = []
    evidence = {}
    terms = set()
    for name, rule in RULES.items():
        lines.append(f"fof({name},axiom,({rule['formula']})).")
        evidence[name] = {"kind": "rule", **rule}
    for number, row in enumerate(records):
        literal = parse_literal(row["kif"])
        terms.update((literal.left, literal.right))
        name = f"axiom_{number}"
        lines.append(f"fof({name},axiom,({literal.tptp()})).")
        evidence[name] = dict(row)
    return "\n".join(lines) + "\n", evidence, terms
