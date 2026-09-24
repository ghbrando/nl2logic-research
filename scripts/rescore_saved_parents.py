"""Re-score saved no-memory declaration outputs by parent likelihood.

For each saved prediction with a declaration, rebuild the exact encoder prompt
(checked against the saved SHA-256), score every offered parent by sequence
log-likelihood, and compare the likelihood choice with the beam output. The
passage-only gate reviews both choices. Labels are joined only from an
existing scored.jsonl after scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.grounding import assess_declared_definition
from src.training.train import format_prompt

DEV_POOL = ["Process", "IntelligenceProduct"]


def rebuild(row: dict) -> tuple[str, list[str]]:
    declaration = row["declaration"]
    parents = row.get("parent_candidates") or DEV_POOL
    prompt = (
        f"{format_prompt(row['extraction']['candidate_text'])}\nCurrent source evidence: {row['evidence_sentence']}\n"
        f"Approved class name only: {declaration['declaration']}\n"
        f"Available existing classes: {', '.join(parents)}.\n"
        "Produce a new claim only if the current source supports it."
    )
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row["arms"]["baseline"]["model_prompt_sha256"]:
        raise ValueError(f"{row['record_id']}: rebuilt prompt differs from the saved run")
    return prompt, list(parents)


def run(runs: list[tuple[Path, Path]], sources: list[Path], model_path: Path, output_dir: Path) -> dict:
    import torch
    from src.eval.model_predictor import load_model_and_tokenizer

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    def logp(prompt: str, target: str) -> float:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        labels = tokenizer(target, return_tensors="pt").input_ids
        with torch.no_grad():
            return float(-model(**encoded, labels=labels).loss * labels.shape[1])

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for (predictions, scored), source_path in zip(runs, sources):
        excerpts = {r["record_id"]: r["source_excerpt"]
                    for r in map(json.loads, source_path.read_text(encoding="utf-8").splitlines())}
        gold = {}
        for line in scored.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["mode"] == "baseline":
                gold[r["record_id"]] = r["gold_kif_exact"] if r["label_decision"] == "positive" else None
        for line in predictions.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["declaration"] or "baseline" not in row["arms"]:
                continue
            prompt, parents = rebuild(row)
            symbol = row["declaration"]["symbol"]
            scores = {p: logp(prompt, f"{symbol} subclass-of {p}") for p in parents}
            choice = max(scores, key=scores.get)
            beam = row["arms"]["baseline"]["cnl"]

            def gate(cnl):
                return assess_declared_definition(
                    cnl, passage=excerpts[row["record_id"]], evidence_sentence=row["evidence_sentence"],
                    declaration=row["declaration"], parent_terms=parents).accepted if cnl else False

            accepted_parents = [p for p in parents if gate(f"{symbol} subclass-of {p}")]
            rows.append({
                "run": predictions.parent.name, "record_id": row["record_id"], "paragraph": row["paragraph"],
                "parents": parents, "logp": scores, "likelihood_cnl": f"{symbol} subclass-of {choice}",
                "beam_cnl": beam, "gate_accepted_parents": accepted_parents,
                "likelihood_gate_accepted": gate(f"{symbol} subclass-of {choice}"), "beam_gate_accepted": gate(beam),
                "beam_label_match": gold.get(row["record_id"]),
            })
    with (output_dir / "items.jsonl").open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    stated = [r for r in rows if len(r["gate_accepted_parents"]) == 1]
    summary = {
        "declared_items": len(rows),
        "likelihood_equals_beam": sum(r["likelihood_cnl"] == r["beam_cnl"] for r in rows),
        "likelihood_choice_counts": {p: sum(r["likelihood_cnl"].endswith(f" {p}") for r in rows)
                                     for p in sorted({p for r in rows for p in r["parents"]})},
        "items_with_one_passage_stated_parent": len(stated),
        "likelihood_picks_stated_parent": sum(r["likelihood_cnl"].split()[-1] == r["gate_accepted_parents"][0] for r in stated),
        "beam_picks_stated_parent": sum((r["beam_cnl"] or "").split()[-1:] == r["gate_accepted_parents"] for r in stated),
        "stated_parent_items": [{k: r[k] for k in ("run", "paragraph", "gate_accepted_parents", "likelihood_cnl", "beam_cnl")}
                                for r in stated],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "interpretation": "Development re-scoring of already-queried outputs; the gate still decides acceptance.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", metavar=("PREDICTIONS", "SCORED", "SOURCES"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run([(Path(p), Path(s)) for p, s, _ in args.run], [Path(src) for _, _, src in args.run],
        args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
