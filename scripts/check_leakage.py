from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.evaluate import DEFAULT_GOLD_PATH
from src.training.train import DEFAULT_PREPARED_TRAIN_FILE, PATTERN_COVERAGE_SENTENCES


@dataclass(frozen=True)
class LeakageReport:
    training_path: Path
    gold_path: Path
    nl_matches: list[tuple[str, str]]
    cnl_matches: list[tuple[str, str]]
    probe_matches: list[tuple[str, str]]

    @property
    def has_leakage(self) -> bool:
        return bool(self.nl_matches or self.cnl_matches or self.probe_matches)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check prepared training data for exact leakage against gold eval and pattern probes."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_PREPARED_TRAIN_FILE,
        help=f"Prepared training JSONL path (default: {DEFAULT_PREPARED_TRAIN_FILE})",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Gold evaluation JSONL path (default: {DEFAULT_GOLD_PATH})",
    )
    parser.add_argument(
        "--ignore-cnl-overlap",
        action="store_true",
        help="Ignore exact gold CNL matches and only fail on exact gold NL or probe NL overlaps.",
    )
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def check_leakage(
    *,
    training_path: Path = DEFAULT_PREPARED_TRAIN_FILE,
    gold_path: Path = DEFAULT_GOLD_PATH,
    check_gold_cnl_overlap: bool = True,
) -> LeakageReport:
    training_records = _load_jsonl(training_path)
    gold_records = _load_jsonl(gold_path)

    training_nls = {record["nl"] for record in training_records if "nl" in record}
    training_cnls = {record["cnl"] for record in training_records if "cnl" in record}
    gold_nls = {record["nl"] for record in gold_records if "nl" in record}
    gold_cnls = {record["cnl"] for record in gold_records if "cnl" in record}

    nl_matches = [("nl", value) for value in sorted(training_nls & gold_nls)]
    cnl_matches = [("cnl", value) for value in sorted(training_cnls & gold_cnls)] if check_gold_cnl_overlap else []

    probe_matches: list[tuple[str, str]] = []
    for pattern, nl in PATTERN_COVERAGE_SENTENCES:
        if nl in training_nls:
            probe_matches.append((pattern, nl))

    return LeakageReport(
        training_path=training_path,
        gold_path=gold_path,
        nl_matches=nl_matches,
        cnl_matches=cnl_matches,
        probe_matches=probe_matches,
    )


def print_report(report: LeakageReport) -> None:
    print("Leakage check:")
    print(f"  Training artifact: {report.training_path}")
    print(f"  Gold set:          {report.gold_path}")

    if not report.has_leakage:
        print("  Status: clean")
        return

    print("  Status: leakage detected")
    if report.nl_matches:
        print("  Exact NL matches:")
        for _, value in report.nl_matches:
            print(f"    - {value}")
    if report.cnl_matches:
        print("  Exact CNL matches:")
        for _, value in report.cnl_matches:
            print(f"    - {value}")
    if report.probe_matches:
        print("  Pattern coverage probe matches:")
        for pattern, value in report.probe_matches:
            print(f"    - [{pattern}] {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_leakage(
        training_path=args.train_path,
        gold_path=args.gold_path,
        check_gold_cnl_overlap=not args.ignore_cnl_overlap,
    )
    print_report(report)
    return 1 if report.has_leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
