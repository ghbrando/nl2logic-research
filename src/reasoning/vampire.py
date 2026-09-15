"""Vampire subprocess boundary. Exit codes alone never establish a theorem."""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProverResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_vampire(executable: str, problem: Path, timeout: int = 10) -> ProverResult:
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    try:
        result = subprocess.run(
            [executable, "--input_syntax", "tptp", "-p", "tptp", "--output_axiom_names", "on", "-t", str(timeout), str(problem.resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=timeout + 2,
        )
    except subprocess.TimeoutExpired:
        return ProverResult("Timeout", stderr="External wall-clock timeout exceeded")
    except OSError as exc:
        return ProverResult("Error", stderr=str(exc))
    statuses = set(re.findall(r"^%\s*SZS status (\w+)\b", result.stdout, re.MULTILINE))
    status = next(iter(statuses)) if len(statuses) == 1 and result.returncode == 0 else "Error"
    return ProverResult(status, result.stdout, result.stderr, result.returncode)


def proof_inputs(output: str, evidence: dict[str, dict]) -> list[str] | None:
    """Read named inputs from the emitted refutation, not echoed input text.

    Retain the complete proof as the audit artifact; this is provenance
    extraction, not independent proof checking or proof minimization.
    """
    match = re.search(r"% SZS output start Proof[^\n]*\n(.*?)% SZS output end Proof", output, re.DOTALL)
    if not match or "$false" not in match.group(1):
        return None
    names = re.findall(r"\bfile\(\s*(?:'[^']*'|[^,]+)\s*,\s*'?([a-z][a-z0-9_]*)'?\s*\)", match.group(1))
    if set(names) - evidence.keys() - {"query"}:
        return None
    used = sorted(set(names) & evidence.keys())
    return used or None
