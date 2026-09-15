"""Shared, label-isolated scoring for sentence-to-CNL comparisons."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Callable

from src.compiler.compiler import CNLCompiler
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.ingest.grounding import DoctrineGrounder


_DOCTRINE_KIF_PATH = Path(__file__).resolve().parents[2] / "data" / "ontology" / "doctrine_domain.kif"
_KIF_SUBCLASS_RE = re.compile(r"\(subclass\s+(\S+)\s+([^\s)]+)\s*\)")
_ONTOLOGY_SUBCLASS_CACHE: set[tuple[str, str]] | None = None


def load_asserted_subclass_pairs(path: Path) -> set[tuple[str, str]]:
    """Every (child, parent) the given ontology file asserts.

    KIF line comments start with `;`. A commented-out assertion is not asserted,
    and counting it would let a disabled claim mask a genuine derivation.
    """
    live = "\n".join(
        line.split(";", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    return set(_KIF_SUBCLASS_RE.findall(live))


def subclass_claims(kif: str) -> list[tuple[str, str]]:
    """Every (child, parent) claim in a KIF output."""
    return _KIF_SUBCLASS_RE.findall(kif or "")


def _asserted_subclass_pairs() -> set[tuple[str, str]]:
    global _ONTOLOGY_SUBCLASS_CACHE
    if _ONTOLOGY_SUBCLASS_CACHE is None:
        _ONTOLOGY_SUBCLASS_CACHE = load_asserted_subclass_pairs(_DOCTRINE_KIF_PATH)
    return _ONTOLOGY_SUBCLASS_CACHE


def restates_ontology(kif: str) -> bool:
    """True when every subclass claim in the output is already asserted in the ontology.

    The doctrine KIF supplies implied parents to both the prompt router and the
    grounding gate, so a claim it already contains can be produced and validated
    without the source sentence carrying it. Such an accept measures background
    knowledge, not formalization, and must not be counted as coverage.

    Outputs with no subclass claim return False: there is nothing to restate.
    """
    claims = _KIF_SUBCLASS_RE.findall(kif or "")
    if not claims:
        return False
    asserted = _asserted_subclass_pairs()
    return all(claim in asserted for claim in claims)


def normalized_nl(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_benchmark(paths: list[Path], compiler=None) -> list[dict]:
    compiler = compiler or CNLCompiler()
    records, seen, seen_texts = [], set(), set()
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rid = row.get("record_id")
            if not isinstance(rid, str) or not rid or rid in seen:
                raise ValueError(f"{path}:{number}: missing/duplicate record_id")
            if not isinstance(row.get("nl"), str) or not row["nl"].strip():
                raise ValueError(f"{rid}: missing source text")
            if not isinstance(row.get("doc_id"), str) or not row["doc_id"]:
                raise ValueError(f"{rid}: document ID is required")
            text_key = (row["doc_id"], normalized_nl(row["nl"]))
            if text_key in seen_texts:
                raise ValueError(f"{rid}: duplicate source text within document")
            if row.get("status") not in {"draft", "reviewed", "accepted"}:
                raise ValueError(f"{rid}: invalid review status")
            reviewed = row["status"] in {"reviewed", "accepted"}
            if reviewed:
                if type(row.get("formalizable")) is not bool:
                    raise ValueError(f"{rid}: reviewed records need a boolean formalizable label")
                if row["formalizable"]:
                    if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("cnl", "kif")):
                        raise ValueError(f"{rid}: reviewed positives need CNL and KIF")
                    try:
                        compiled = compiler.compile(row["cnl"])
                    except Exception as exc:
                        raise ValueError(f"{rid}: invalid gold CNL: {exc}") from exc
                    if compiled.strip() != row["kif"].strip():
                        raise ValueError(f"{rid}: gold CNL/KIF mismatch")
                elif row.get("cnl") is not None or row.get("kif") is not None:
                    raise ValueError(f"{rid}: abstention gold must have null CNL/KIF")
            seen.add(rid)
            seen_texts.add(text_key)
            records.append({**row, "benchmark_file": str(path), "benchmark_line": number})
    if not records:
        raise ValueError("Benchmark is empty")
    return records


def training_overlap(records: list[dict], paths: list[Path]) -> dict[str, list[str]]:
    """Detect normalized exact NL overlap. This is not a semantic leakage test."""
    lookup = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("nl"), str):
                    lookup.setdefault(normalized_nl(row["nl"]), set()).add(str(path))
    return {r["record_id"]: sorted(lookup[normalized_nl(r["nl"])])
            for r in records if normalized_nl(r["nl"]) in lookup}


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def evaluate_method(records: list[dict], predictor: Callable[[str], str | None], *,
                    overlaps: dict | None = None, compiler=None, grounder=None) -> list[dict]:
    compiler = compiler or CNLCompiler()
    grounder = grounder or DoctrineGrounder()
    results = []
    for row in records:
        scored = row["status"] in {"reviewed", "accepted"} and row.get("split") != "train"
        exclusion = None if scored else "unreviewed_or_training_split"
        if row["record_id"] in (overlaps or {}):
            scored, exclusion = False, "training_text_overlap"
        result = {
            "record_id": row["record_id"], "doc_id": row.get("doc_id"),
            "page": row.get("page"), "pdf_page": row.get("pdf_page"),
            "nl": row["nl"], "scored": scored, "exclusion": exclusion,
            "expected_formalizable": row.get("formalizable") if scored else None,
            "gold_kif": row.get("kif") if scored else None,
            "pattern": row.get("pattern"), "phenomena": row.get("phenomena", []),
            "cnl": None, "kif": None, "route": "abstain", "reason": None,
            "error": None, "kif_exact": None, "accepted_mismatch": False,
            "ontology_restated": None,
        }
        start = perf_counter()
        try:
            # Predictors receive source text only, never gold fields.
            if CNLSampler.abstain_if_unsupported(row["nl"]):
                result["reason"] = "shared_input_gate"
            else:
                result["cnl"] = predictor(row["nl"])
                if result["cnl"]:
                    try:
                        result["kif"] = compiler.compile(result["cnl"])
                    except Exception as exc:
                        result.update(route="compile_error", error=str(exc))
                    else:
                        assessment = grounder.assess(
                            cnl=result["cnl"], original_text=row["nl"],
                            normalized_text=row["nl"], linked_text=row["nl"], subclaim_text=row["nl"],
                        )
                        result.update(route="accepted" if assessment.accepted else "review", reason=assessment.reason)
        except UnsupportedInputError as exc:
            result.update(route="abstain", reason=str(exc))
        except Exception as exc:
            result.update(route="error", error=f"{type(exc).__name__}: {exc}")
        if result["kif"]:
            result["ontology_restated"] = restates_ontology(result["kif"])
        result["latency_seconds"] = perf_counter() - start
        if scored:
            result["kif_exact"] = bool(row["formalizable"] and result["kif"] and result["kif"].strip() == row["kif"].strip())
            result["accepted_mismatch"] = result["route"] == "accepted" and not result["kif_exact"]
        results.append(result)
    return results


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if r["scored"]]
    accepted = [r for r in scored if r["route"] == "accepted"]
    positives = [r for r in scored if r["expected_formalizable"]]
    negatives = [r for r in scored if not r["expected_formalizable"]]
    correct = sum(bool(r["kif_exact"]) for r in accepted)
    ratio = lambda n, d: n / d if d else None
    return {
        "total": len(results), "scored": len(scored), "unscored": len(results) - len(scored),
        "routes_all": dict(Counter(r["route"] for r in results)),
        "routes_scored": dict(Counter(r["route"] for r in scored)),
        "accepted_exact_matches": correct,
        "accepted_exact_precision": ratio(correct, len(accepted)),
        "accepted_coverage": ratio(len(accepted), len(scored)),
        "correct_accepted_positive_coverage": ratio(correct, len(positives)),
        "abstention_accuracy": ratio(sum(r["route"] == "abstain" for r in negatives), len(negatives)),
        "false_accepts_on_abstain_gold": sum(r["route"] == "accepted" for r in negatives),
        "accepted_mismatches_needing_adjudication": sum(r["accepted_mismatch"] for r in scored),
        "accepted_restating_ontology": sum(bool(r["ontology_restated"]) for r in accepted),
        "accepted_novel": sum(not r["ontology_restated"] for r in accepted),
        "accepted_novel_coverage": ratio(
            sum(not r["ontology_restated"] for r in accepted), len(scored)
        ),
        "accepted_restating_ontology_all": sum(
            bool(r["ontology_restated"]) for r in results if r["route"] == "accepted"
        ),
        "review_queue_size": sum(r["route"] == "review" for r in results),
        "human_review_seconds": None,
        "total_latency_seconds": sum(r["latency_seconds"] for r in results),
        "metric_limit": (
            "Exact KIF agreement is a proxy, not semantic precision. Review count is "
            "workload, not measured human effort. Accepted coverage includes claims the "
            "doctrine ontology already asserts; accepted_novel_coverage excludes them and "
            "is the figure to compare methods on."
        ),
    }
