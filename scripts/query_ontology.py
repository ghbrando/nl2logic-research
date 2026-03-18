from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = _REPO_ROOT / "data" / "ontology" / "fm2-0_index.json"


class QueryError(Exception):
    """Raised when the ontology query CLI cannot load or query the index."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query accepted ontology facts from an index JSON file.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help=f"Index path (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--term", type=str, default=None, help="Ontology term to match exactly.")
    parser.add_argument("--relation", type=str, default=None, help="Relation term to match exactly.")
    parser.add_argument("--section", type=str, default=None, help="Section label to match exactly.")
    parser.add_argument("--page", type=int, default=None, help="Page number to match exactly.")
    parser.add_argument("--text", type=str, default=None, help="Case-insensitive substring search across source text and formalizations.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum results to print (default: 20)")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON results.")
    return parser.parse_args(argv)


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise QueryError(f"Index file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QueryError(f"Malformed index JSON: {path}") from exc


def _candidate_ids(index: dict[str, Any], *, relation: str | None, term: str | None, section: str | None, page: int | None) -> set[str] | None:
    candidate_sets: list[set[str]] = []
    if relation is not None:
        candidate_sets.append(set(index.get("by_relation", {}).get(relation, [])))
    if term is not None:
        candidate_sets.append(set(index.get("by_term", {}).get(term, [])))
    if section is not None:
        candidate_sets.append(set(index.get("by_section", {}).get(section, [])))
    if page is not None:
        candidate_sets.append(set(index.get("by_page", {}).get(str(page), [])))
    if not candidate_sets:
        return None
    current = candidate_sets[0]
    for candidate_set in candidate_sets[1:]:
        current &= candidate_set
    return current


def query_records(
    index: dict[str, Any],
    *,
    term: str | None = None,
    relation: str | None = None,
    section: str | None = None,
    page: int | None = None,
    text: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    records = index.get("records", [])
    candidate_ids = _candidate_ids(index, relation=relation, term=term, section=section, page=page)
    text_query = text.casefold() if text is not None else None
    matches: list[dict[str, Any]] = []

    for record in records:
        if candidate_ids is not None and record.get("record_id") not in candidate_ids:
            continue
        if text_query is not None:
            haystack = " ".join(
                str(record.get(field, ""))
                for field in ("nl", "original", "cnl", "kif", "section", "relation")
            ).casefold()
            if text_query not in haystack:
                continue
        matches.append(record)
        if len(matches) >= max(0, limit):
            break

    return matches


def format_record(record: dict[str, Any], ordinal: int) -> str:
    section = record.get("section") or "(no section)"
    relation = record.get("relation") or "(no relation)"
    return (
        f"[{ordinal}] {record.get('record_id', '<unknown>')} | page {record.get('page_number')} | {section}\n"
        f"Relation: {relation}\n"
        f"NL: {record.get('nl', '')}\n"
        f"CNL: {record.get('cnl', '')}\n"
        f"KIF: {record.get('kif', '')}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = load_index(args.index)
    matches = query_records(
        index,
        term=args.term,
        relation=args.relation,
        section=args.section,
        page=args.page,
        text=args.text,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(matches, ensure_ascii=False, indent=2))
        return 0
    if not matches:
        print("No matching accepted facts found.")
        return 0
    for ordinal, record in enumerate(matches, start=1):
        if ordinal > 1:
            print()
        print(format_record(record, ordinal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
