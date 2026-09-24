"""Does the constrained model's parent choice follow the passage's stated genus?

Synthetic minimal pairs, no labels and no fresh data: each defined phrase from
existing registries is given two definitions that differ only in genus ("the
process of ..." versus "intelligence that is produced from ..."). The model
scores both parent candidates by sequence log-likelihood, raw and calibrated
against the same prompt with the evidence replaced by a genus-free sentence.
A model that ignores the passage cannot get both members of a pair right.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.grounding import assess_declared_definition
from src.training.train import format_prompt

PARENTS = ("Process", "IntelligenceProduct")
VARIANTS = {
    "Process": ("is the process", "is the process of collecting and analyzing information for commanders."),
    "IntelligenceProduct": ("is intelligence", "is intelligence that is produced from collected information for commanders."),
}
NEUTRAL = "is described in this publication for commanders."


def _pascal(text: str) -> str:
    return "".join(word[:1].upper() + word[1:].lower() for word in re.findall(r"[A-Za-z0-9]+", text))


def load_aliases(registry_paths: list[Path]) -> list[str]:
    aliases: dict[str, str] = {}
    for path in registry_paths:
        for entry in json.loads(path.read_text(encoding="utf-8"))["declarations"]:
            key = " ".join(entry["alias"].casefold().split())
            if re.fullmatch(r"[a-z][a-z0-9 '-]*", key) and len(key.split()) <= 6:
                aliases.setdefault(key, entry["alias"])
    return [aliases[key] for key in sorted(aliases)]


def build_items(aliases: list[str]) -> list[dict]:
    """Two definitions per alias; the gate confirms each variant's stated parent."""
    items = []
    for alias in aliases:
        subject = alias[:1].upper() + alias[1:]
        symbol = _pascal(alias)
        for parent, (cue, tail) in VARIANTS.items():
            sentence = f"{subject} {tail}"
            declaration = {"alias": alias, "evidence_cue": cue, "symbol": symbol}
            verdicts = {
                candidate: assess_declared_definition(
                    f"{symbol} subclass-of {candidate}", passage=sentence, evidence_sentence=sentence,
                    declaration=declaration, parent_terms=PARENTS,
                ).accepted
                for candidate in PARENTS
            }
            if verdicts != {candidate: candidate == parent for candidate in PARENTS}:
                raise ValueError(f"Gate does not isolate the stated parent for: {sentence}")
            items.append({"alias": alias, "symbol": symbol, "stated_parent": parent, "sentence": sentence,
                          "candidate_text": f"{subject} {cue}"})
    return items


def model_prompt(candidate_text: str, evidence: str, symbol: str) -> str:
    # Same layout as scripts/run_partial_claim_declaration_probe.py.
    return (
        f"{format_prompt(candidate_text)}\nCurrent source evidence: {evidence}\n"
        f"Approved class name only: (instance {symbol} Class)\n"
        f"Available existing classes: {', '.join(PARENTS)}.\n"
        "Produce a new claim only if the current source supports it."
    )


def run(registry_paths: list[Path], model_path: Path, output_dir: Path) -> dict:
    import torch
    from src.eval.model_predictor import load_model_and_tokenizer

    if torch.cuda.is_available():
        raise RuntimeError("Discrimination probe is CPU-only")
    torch.set_num_threads(2)
    items = build_items(load_aliases(registry_paths))
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    def logp(prompt: str, target: str) -> float:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        labels = tokenizer(target, return_tensors="pt").input_ids
        with torch.no_grad():
            loss = model(**encoded, labels=labels).loss
        return float(-loss * labels.shape[1])

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in items:
        subject = item["alias"][:1].upper() + item["alias"][1:]
        raw = {p: logp(model_prompt(item["candidate_text"], item["sentence"], item["symbol"]),
                       f"{item['symbol']} subclass-of {p}") for p in PARENTS}
        null = {p: logp(model_prompt(item["candidate_text"], f"{subject} {NEUTRAL}", item["symbol"]),
                        f"{item['symbol']} subclass-of {p}") for p in PARENTS}
        calibrated = {p: raw[p] - null[p] for p in PARENTS}
        rows.append(item | {"raw_logp": raw, "null_logp": null,
                            "raw_choice": max(raw, key=raw.get), "calibrated_choice": max(calibrated, key=calibrated.get)})
    with (output_dir / "items.jsonl").open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)

    def pair_accuracy(key: str) -> dict:
        by_alias: dict[str, list[bool]] = {}
        for row in rows:
            by_alias.setdefault(row["alias"], []).append(row[key] == row["stated_parent"])
        return {
            "item_accuracy": sum(row[key] == row["stated_parent"] for row in rows) / len(rows),
            "both_in_pair_correct": sum(all(v) for v in by_alias.values()) / len(by_alias),
            "choice_counts": {p: sum(row[key] == p for row in rows) for p in PARENTS},
        }

    summary = {
        "aliases": len({row["alias"] for row in rows}),
        "items": len(rows),
        "raw": pair_accuracy("raw_choice"),
        "calibrated": pair_accuracy("calibrated_choice"),
        "chance_both_in_pair_for_constant_choice": 0.0,
        "registries": [path.as_posix() for path in registry_paths],
        "adapter_sha256": hashlib.sha256((model_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "python": platform.python_version(), "device": "cpu",
        "interpretation": "Synthetic development diagnostic of passage conditioning; not accuracy on doctrine.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registries", type=Path, nargs="+", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.registries, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
