"""Score a trained sense-veto adapter on synthetic held-out and development claims.

Decision rule, fixed before evaluation: veto when log p(not claim) exceeds
log p(claim). Development claims are every claim the passage-only gate
accepted in the listed scored runs; labels are only joined afterward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.sense_veto import accept_target, reject_target, veto_prompt
from src.training.train import format_prompt

_KIF = re.compile(r"\(subclass ([A-Za-z0-9_]+) ([A-Za-z0-9_]+)\)")


def dev_claims(scored_paths: list[Path]) -> list[dict]:
    claims = {}
    for path in scored_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("declared_gate_accepted") is not True:
                continue
            child, parent = _KIF.fullmatch(row["compiled_kif"]).groups()
            run_name = path.parent.parent.name
            key = (run_name, row["record_id"], row["compiled_kif"])
            entry = claims.setdefault(key, {
                "run": run_name, "record_id": row["record_id"], "paragraph": row["paragraph"],
                "sentence": row["evidence_sentence"], "symbol": child, "parent": parent,
                "label_match": row["gold_kif_exact"], "modes": []})
            entry["modes"].append(row["mode"])
    return list(claims.values())


def run(adapter: Path, synthetic: list[Path], scored: list[Path], output_dir: Path) -> dict:
    import torch
    from src.eval.model_predictor import load_model_and_tokenizer

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(adapter)
    model.eval()

    def logp(prompt: str, target: str) -> float:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        labels = tokenizer(target, return_tensors="pt").input_ids
        with torch.no_grad():
            return float(-model(**encoded, labels=labels).loss * labels.shape[1])

    def judge(sentence: str, symbol: str, parent: str) -> dict:
        prompt = format_prompt(veto_prompt(sentence, symbol, parent))
        keep, veto = logp(prompt, accept_target(symbol, parent)), logp(prompt, reject_target(symbol, parent))
        return {"logp_accept": keep, "logp_reject": veto, "veto": veto > keep}

    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "decision_rule": "veto if logp(not claim) > logp(claim)"}
    for path in synthetic:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row.update(judge(row["sentence"], row["symbol"], row["parent"]))
        with (output_dir / f"{path.stem}.scored.jsonl").open("x", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in rows)
        by_parent = {}
        for row in rows:
            stats = by_parent.setdefault(row["parent"], {"accept_kept": 0, "accept_total": 0, "reject_vetoed": 0, "reject_total": 0})
            if row["label"] == "accept":
                stats["accept_total"] += 1
                stats["accept_kept"] += not row["veto"]
            else:
                stats["reject_total"] += 1
                stats["reject_vetoed"] += row["veto"]
        summary[path.stem] = {
            "accept_kept": sum(s["accept_kept"] for s in by_parent.values()),
            "accept_total": sum(s["accept_total"] for s in by_parent.values()),
            "reject_vetoed": sum(s["reject_vetoed"] for s in by_parent.values()),
            "reject_total": sum(s["reject_total"] for s in by_parent.values()),
            "by_parent": by_parent,
        }
    claims = dev_claims(scored)
    for claim in claims:
        claim.update(judge(claim["sentence"], claim["symbol"], claim["parent"]))
    with (output_dir / "development_claims.jsonl").open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(c) + "\n" for c in claims)
    summary["development"] = {
        "gate_accepted_claims": len(claims),
        "label_matches_kept": sum(c["label_match"] and not c["veto"] for c in claims),
        "label_matches_total": sum(bool(c["label_match"]) for c in claims),
        "unsupported_vetoed": sum(not c["label_match"] and c["veto"] for c in claims),
        "unsupported_total": sum(not c["label_match"] for c in claims),
        "claims": [{k: c[k] for k in ("run", "paragraph", "symbol", "parent", "label_match", "modes", "veto")} for c in claims],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True)
    parser.add_argument("--scored", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.adapter, args.synthetic, args.scored, args.output_dir)


if __name__ == "__main__":
    main()
