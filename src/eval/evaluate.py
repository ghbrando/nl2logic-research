"""
Gold-set evaluation for NL2Logic.

The default CLI mode uses the gold CNL strings as an oracle baseline after
first recompiling them through CNLCompiler to verify that the gold set itself
is internally clean. The same scoring path also supports an injected predictor
for real NL -> CNL evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.compiler.compiler import CNLCompiler

DEFAULT_GOLD_PATH = _REPO_ROOT / "data" / "eval" / "gold_pairs.jsonl"


@dataclass(frozen=True)
class GoldPair:
    nl: str
    cnl: str
    kif: str
    pattern: str


@dataclass(frozen=True)
class PatternMetrics:
    total: int
    cnl_exact: int
    kif_exact: int
    abstained: int

    @property
    def cnl_accuracy(self) -> float:
        return self.cnl_exact / self.total if self.total else 0.0

    @property
    def kif_accuracy(self) -> float:
        return self.kif_exact / self.total if self.total else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    cnl_exact: int
    kif_exact: int
    abstained: int
    pattern_breakdown: dict[str, PatternMetrics]

    @property
    def cnl_accuracy(self) -> float:
        return self.cnl_exact / self.total if self.total else 0.0

    @property
    def kif_accuracy(self) -> float:
        return self.kif_exact / self.total if self.total else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NL2Logic on a gold NL/CNL/KIF set.")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Gold JSONL path (default: {DEFAULT_GOLD_PATH})",
    )
    return parser.parse_args(argv)


def load_gold_pairs(path: Path = DEFAULT_GOLD_PATH) -> list[GoldPair]:
    if not path.exists():
        raise FileNotFoundError(f"Gold evaluation file not found: {path}")

    pairs: list[GoldPair] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"nl", "cnl", "kif", "pattern"}
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"Line {line_number} in {path} is missing required fields: {', '.join(sorted(missing))}"
                )
            pairs.append(
                GoldPair(
                    nl=record["nl"],
                    cnl=record["cnl"],
                    kif=record["kif"],
                    pattern=record["pattern"],
                )
            )

    if not pairs:
        raise ValueError(f"No gold evaluation pairs loaded from {path}")
    return pairs


def validate_gold_pairs(
    pairs: list[GoldPair],
    *,
    compiler: CNLCompiler | None = None,
) -> None:
    compiler = compiler or CNLCompiler()
    for pair in pairs:
        compiled = compiler.compile(pair.cnl)
        if compiled != pair.kif:
            raise ValueError(
                "Gold KIF mismatch for NL "
                f"{pair.nl!r}: compiler produced {compiled!r}, stored {pair.kif!r}"
            )


def _oracle_predictor(pair: GoldPair) -> str:
    return pair.cnl


def _predict_cnl(pair: GoldPair, predictor: Callable[[GoldPair], str | None] | None) -> str | None:
    if predictor is None:
        return _oracle_predictor(pair)
    return predictor(pair)


def evaluate_pairs(
    pairs: list[GoldPair],
    *,
    predictor: Callable[[GoldPair], str | None] | None = None,
    compiler: CNLCompiler | None = None,
) -> EvaluationReport:
    compiler = compiler or CNLCompiler()
    validate_gold_pairs(pairs, compiler=compiler)

    pattern_totals: dict[str, dict[str, int]] = {}
    cnl_exact = 0
    kif_exact = 0
    abstained = 0

    for pair in pairs:
        pattern_counts = pattern_totals.setdefault(
            pair.pattern,
            {"total": 0, "cnl_exact": 0, "kif_exact": 0, "abstained": 0},
        )
        pattern_counts["total"] += 1

        predicted_cnl = _predict_cnl(pair, predictor)
        if predicted_cnl is None:
            abstained += 1
            pattern_counts["abstained"] += 1
            continue

        predicted_cnl = predicted_cnl.strip()
        predicted_kif: str | None
        try:
            predicted_kif = compiler.compile(predicted_cnl)
        except Exception:
            predicted_kif = None

        if predicted_cnl == pair.cnl:
            cnl_exact += 1
            pattern_counts["cnl_exact"] += 1
        if predicted_kif == pair.kif:
            kif_exact += 1
            pattern_counts["kif_exact"] += 1

    breakdown = {
        pattern: PatternMetrics(
            total=counts["total"],
            cnl_exact=counts["cnl_exact"],
            kif_exact=counts["kif_exact"],
            abstained=counts["abstained"],
        )
        for pattern, counts in sorted(pattern_totals.items())
    }

    return EvaluationReport(
        total=len(pairs),
        cnl_exact=cnl_exact,
        kif_exact=kif_exact,
        abstained=abstained,
        pattern_breakdown=breakdown,
    )


def print_report(report: EvaluationReport) -> None:
    print("Gold evaluation:")
    print(f"  Total pairs:             {report.total}")
    print(f"  Exact match CNL accuracy {report.cnl_accuracy:.3f} ({report.cnl_exact}/{report.total})")
    print(f"  Exact match KIF accuracy {report.kif_accuracy:.3f} ({report.kif_exact}/{report.total})")
    print(f"  Abstain rate:            {report.abstain_rate:.3f} ({report.abstained}/{report.total})")
    print("\nPattern breakdown:")
    for pattern, metrics in report.pattern_breakdown.items():
        print(
            f"  {pattern:12s} total={metrics.total:2d} "
            f"cnl={metrics.cnl_accuracy:.3f} "
            f"kif={metrics.kif_accuracy:.3f} "
            f"abstain={metrics.abstain_rate:.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs = load_gold_pairs(args.gold_path)
    report = evaluate_pairs(pairs)
    print_report(report)


if __name__ == "__main__":
    main()
