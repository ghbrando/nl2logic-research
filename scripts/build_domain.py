from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_kif.jsonl"
DEFAULT_THEORY_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_domain.kif"
DEFAULT_METADATA_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_domain_metadata.json"


class DomainBuildError(Exception):
    """Raised when accepted doctrine facts cannot be assembled into a domain bundle."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a prover-ready FM 2-0 doctrine domain bundle.")
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED_PATH, help=f"Accepted KIF JSONL path (default: {DEFAULT_ACCEPTED_PATH})")
    parser.add_argument("--theory-out", type=Path, default=DEFAULT_THEORY_PATH, help=f"Output theory file path (default: {DEFAULT_THEORY_PATH})")
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_PATH, help=f"Output provenance metadata JSON path (default: {DEFAULT_METADATA_PATH})")
    parser.add_argument("--sumo", action="append", default=[], help="Path to a referenced SUMO theory file. May be repeated.")
    parser.add_argument("--vampire", type=str, default=None, help="Optional Vampire executable/command to invoke for validation.")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code if optional validation fails.")
    return parser.parse_args(argv)


def load_accepted_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DomainBuildError(f"Accepted-facts file not found: {path}")
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DomainBuildError(f"Malformed accepted-facts JSONL at line {line_number}: {path}") from exc
            record_id = payload.get("record_id")
            kif = payload.get("kif")
            if not isinstance(record_id, str) or not record_id:
                raise DomainBuildError(f"Missing record_id at line {line_number}: {path}")
            if record_id in seen_record_ids:
                raise DomainBuildError(f"Duplicate record_id in accepted facts: {record_id}")
            if not isinstance(kif, str) or not kif.strip():
                raise DomainBuildError(f"Missing kif at line {line_number}: {path}")
            seen_record_ids.add(record_id)
            records.append(payload)
    return records


def build_metadata(records: list[dict[str, Any]], sumo_paths: list[str]) -> dict[str, Any]:
    return {
        "sumo_sources": sumo_paths,
        "record_count": len(records),
        "records": {
            record["record_id"]: {
                "page_number": record.get("page_number"),
                "position": record.get("position"),
                "section": record.get("section", ""),
                "original": record.get("original", ""),
                "nl": record.get("nl", ""),
                "cnl": record.get("cnl", ""),
                "relation": record.get("relation"),
                "terms": record.get("terms", []),
            }
            for record in records
        },
    }


def render_theory(records: list[dict[str, Any]], sumo_paths: list[str]) -> str:
    lines = [
        ";;; Auto-generated FM 2-0 doctrine domain bundle",
        f";;; Accepted record count: {len(records)}",
    ]
    if sumo_paths:
        lines.append(";;; Referenced SUMO sources:")
        lines.extend(f";;;   {path}" for path in sumo_paths)
    lines.append("")

    for record in records:
        lines.append(
            f";;; {record['record_id']} | page {record.get('page_number')} | section {record.get('section', '')}"
        )
        lines.append(record["kif"].strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_domain_bundle(
    records: list[dict[str, Any]],
    *,
    theory_path: Path,
    metadata_path: Path,
    sumo_paths: list[str],
) -> None:
    theory_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    theory_path.write_text(render_theory(records, sumo_paths), encoding="utf-8")
    metadata_path.write_text(json.dumps(build_metadata(records, sumo_paths), ensure_ascii=False, indent=2), encoding="utf-8")


def run_vampire_validation(vampire_command: str, theory_path: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [vampire_command, str(theory_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DomainBuildError(f"Failed to invoke Vampire command '{vampire_command}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_accepted_records(args.accepted)
    write_domain_bundle(
        records,
        theory_path=args.theory_out,
        metadata_path=args.metadata_out,
        sumo_paths=[str(Path(path)) for path in args.sumo],
    )

    if args.vampire is None:
        print(f"Wrote theory bundle: {args.theory_out}")
        print(f"Wrote metadata: {args.metadata_out}")
        return 0

    result = run_vampire_validation(args.vampire, args.theory_out)
    print(f"Wrote theory bundle: {args.theory_out}")
    print(f"Wrote metadata: {args.metadata_out}")
    print(f"Vampire exit code: {result.returncode}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if args.strict and result.returncode != 0:
        return result.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
