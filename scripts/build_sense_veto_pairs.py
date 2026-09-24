"""Build synthetic sense-veto training and held-out pairs.

Every item is a made-up definition that the passage-only gate accepts for its
parent, so the veto learns exactly the decision the gate cannot make: whether
the genus word is used in the parent class's sense ("a device that measures
..." versus "a device for persuading ..."). Accept items target the claim and
reject items its CNL negation. Subjects come from synthetic word lists, never
doctrine text.

Product and Cycle, the false friends found on development data, never appear
in training; they form a separate held-out-class set that tests transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.grounding import assess_declared_definition
from src.ingest.sense_veto import accept_target, reject_target, veto_prompt
from src.ontology.vocab import load_closed_class_terms
from src.preprocessing.declarations import parent_candidates, propose_declaration
from src.preprocessing.partial_claims import extract_paragraph_candidate

POOL = ["Process", "IntelligenceProduct"]
# (parent, accept templates, reject templates); {S} subject, {x} filler.
TRAIN_SENSES = [
    ("Process", ["{S} is the process of {x} reports for commanders.", "{S} is a process for {x} requirements."], []),
    ("IntelligenceProduct", ["{S} is intelligence that is produced from {x} reports."], []),
    ("Device", ["{S} is a device used to measure {x} signals.", "{S} is a device that detects {x} emissions."],
     ["{S} is a device for persuading an audience during {x} briefings.", "{S} is a device that writers use to hold attention in {x} stories."]),
    ("Vehicle", ["{S} is a vehicle that transports {x} cargo on roads.", "{S} is a vehicle used to carry {x} crews."],
     ["{S} is a vehicle for spreading {x} ideas among the staff.", "{S} is a vehicle for expressing {x} concerns to leaders."]),
    ("Weapon", ["{S} is a weapon that fires {x} projectiles.", "{S} is a weapon used to destroy {x} targets."],
     ["{S} is a weapon for defeating {x} rumors in public debate.", "{S} is a weapon that leaders use against {x} complacency."]),
    ("Map", ["{S} is a map that shows {x} terrain and roads.", "{S} is a map of {x} coastlines and rivers."],
     ["{S} is a map of relationships between {x} tasks.", "{S} is a map of the ideas behind {x} planning."]),
    ("Key", ["{S} is a key that opens the {x} vault lock.", "{S} is a key used to unlock {x} doors."],
     ["{S} is the key to {x} success in planning.", "{S} is the key to understanding {x} behavior."]),
    ("Meeting", ["{S} is a meeting of commanders to plan {x} operations.", "{S} is a meeting of staff officers to discuss {x} tasks."],
     ["{S} is a meeting of two {x} roads at the border.", "{S} is a meeting of {x} rivers below the ridge."]),
    ("Report", ["{S} is a report that describes {x} findings.", "{S} is a report that summarizes {x} results."],
     ["{S} is a report of {x} gunfire heard at night.", "{S} is a report of {x} thunder from the valley."]),
    ("Plan", ["{S} is a plan for moving {x} convoys.", "{S} is a plan for sequencing {x} tasks."],
     ["{S} is a plan of the {x} building floor drawn to scale.", "{S} is a plan of the {x} site drawn from above."]),
    ("Organization", ["{S} is an organization whose members coordinate {x} support.", "{S} is an organization that employs {x} specialists."],
     ["{S} is the organization of {x} files in a cabinet.", "{S} is the organization of {x} tasks into a sequence."]),
    ("Document", ["{S} is a document that records {x} decisions."], []),
    ("Message", ["{S} is a message that informs {x} units of changes."], []),
    ("Database", ["{S} is a database that stores {x} records."], []),
]
# Sensor is omitted: its SUMO gloss covers only software sensors.
HELDOUT_CLASS_SENSES = [
    ("Product", ["{S} is a product that the {x} depot manufactures for sale.", "{S} is a product that factories make from {x} steel."],
     ["{S} is the product of analyzing {x} reports.", "{S} is the product of combining {x} judgments."]),
    ("Cycle", ["{S} is a cycle with two wheels that {x} riders pedal.", "{S} is a cycle that {x} couriers pedal on roads."],
     ["{S} is a cycle of {x} meetings and reports.", "{S} is a cycle of {x} planning and review."]),
]
TRAIN_WORDS = (["relay", "forward", "joint", "regional", "signal", "cargo", "harbor", "survey", "coastal", "liaison",
                "patrol", "supply", "archive", "border", "convoy", "depot"],
               ["audit", "review", "exchange", "screening", "tasking", "handoff", "tracking", "summary", "network",
                "record", "roster", "station", "board", "packet"])
HELDOUT_WORDS = (["harvest", "mountain", "river", "winter", "canal", "orchard", "granite", "summit"],
                 ["census", "ledger", "manifest", "digest", "registry", "inventory", "beacon", "atlas"])
FILLERS = ["field", "local", "coastal", "night", "urban", "remote", "weekly", "border"]


def build(senses, words, *, seed: int, per_template: int) -> list[dict]:
    known = load_closed_class_terms()
    rng = random.Random(seed)
    subjects = [f"{m} {n}" for m in words[0] for n in words[1]]
    rows = []
    for parent, accepts, rejects in senses:
        for label, templates in (("accept", accepts), ("reject", rejects)):
            for template in templates:
                made = 0
                for subject in rng.sample(subjects, len(subjects)):
                    if made == per_template:
                        break
                    sentence = template.format(S=subject[:1].upper() + subject[1:], x=rng.choice(FILLERS))
                    candidate, _ = extract_paragraph_candidate(sentence, sentence)
                    if candidate.status != "candidate" or candidate.gate_reasons:
                        continue
                    declaration = propose_declaration(candidate.candidate_text, known, sentence)
                    if declaration is None:
                        # Article-less genera ("is intelligence that ...") get the
                        # development naming convention: subject plus parent head.
                        stem = "".join(w[:1].upper() + w[1:] for w in subject.split())
                        copula = candidate.candidate_text[len(subject):].split()
                        declaration = {"alias": subject, "evidence_cue": " ".join(copula[:-1]) or copula[0],
                                       "symbol": stem + re.findall(r"[A-Z][a-z0-9]*", parent)[-1],
                                       "genus_phrase": copula[-1].lower()}
                        declaration["declaration"] = f"(instance {declaration['symbol']} Class)"
                    parents = parent_candidates(declaration, POOL, known)
                    if parent not in parents:
                        parents = [*parents, parent]
                    symbol = declaration["symbol"]
                    if not assess_declared_definition(f"{symbol} subclass-of {parent}", passage=sentence,
                                                      evidence_sentence=sentence, declaration=declaration,
                                                      parent_terms=parents).accepted:
                        raise ValueError(f"Gate rejects synthetic item, fix the template: {sentence} -> {parent}")
                    target = accept_target(symbol, parent) if label == "accept" else reject_target(symbol, parent)
                    rows.append({"nl": veto_prompt(sentence, symbol, parent), "cnl": target,
                                 "pattern": f"sense_{label}", "label": label, "parent": parent,
                                 "sentence": sentence, "symbol": symbol})
                    made += 1
                if made < per_template:
                    raise ValueError(f"Too few subjects for template: {template}")
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-template", type=int, default=60)
    parser.add_argument("--heldout-per-template", type=int, default=20)
    args = parser.parse_args()
    train_rows = build(TRAIN_SENSES, TRAIN_WORDS, seed=7, per_template=args.per_template)
    heldout_rows = build(TRAIN_SENSES, HELDOUT_WORDS, seed=11, per_template=args.heldout_per_template)
    heldout_class_rows = build(HELDOUT_CLASS_SENSES, HELDOUT_WORDS, seed=13, per_template=args.heldout_per_template)
    if {r["symbol"] for r in train_rows} & {r["symbol"] for r in heldout_rows + heldout_class_rows}:
        raise ValueError("Held-out subjects overlap training")
    if {r["parent"] for r in train_rows} & {"Product", "Cycle"}:
        raise ValueError("Held-out false-friend classes leaked into training")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"note": "Synthetic subjects only; every item passes the passage-only gate for its parent.",
                "heldout_classes": ["Product", "Cycle"]}
    for name, rows in (("sense_veto_train.jsonl", train_rows), ("sense_veto_heldout.jsonl", heldout_rows),
                       ("sense_veto_heldout_classes.jsonl", heldout_class_rows)):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (args.output_dir / name).write_text(text, encoding="utf-8", newline="\n")
        manifest[name] = {"rows": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                          "labels": {k: sum(r["label"] == k for r in rows) for k in ("accept", "reject")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
