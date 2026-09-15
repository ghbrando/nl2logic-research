"""Draft expressibility triage for a benchmark review packet.

Answers one question before any training run: how much of a real doctrine
chapter can the supported CNL fragment (src/compiler/cnl.lark) express at all?

The verdicts below are a MACHINE DRAFT produced by reading each sentence against
the grammar. They are not gold labels, they are not semantic review, and nothing
here is consumable by scripts/finalize_benchmark_review.py. Their purpose is to
size the grammar's ceiling and to give a human reviewer a starting point to
confirm or overturn, not to replace that review.

A verdict describes the GRAMMAR's reach, not any model's output:

  expressible - a supported pattern carries the sentence's main claim without
                distorting it.
  partial     - a supported pattern carries a weaker true claim; identified
                content is dropped. Acceptable only if the loss is recorded.
  blocked     - no supported pattern carries the main claim without asserting
                something the source does not say.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Blocker codes: what the supported fragment cannot state.
BLOCKERS = {
    "MODALITY": "must / should / may / can / is responsible for - no deontic or alethic operator in the grammar",
    "DISJUNCTION": "or / whether-or inside the claim - the grammar has conjunction only",
    "QUANTIFIER": "most / many / often / generally / primarily, or a comparative or superlative - the grammar has only every/some/no",
    "ATTRIBUTE": "adjectival or literal values (timely, accurate, well-trained) - terms are class names or variables only",
    "EVALUATIVE": "vague evaluative predicate (crucial, difficult, significant, important) with no formal content",
    "TEMPORAL": "continuity, ordering, or aspect (continuously, throughout, in advance of, currently)",
    "SCOPED_NEGATION": "negation inside a restrictive clause - not applies to a whole atomic assertion only",
    "CONDITION": "a condition the grammar cannot state (gerund clause, when appropriate, when available)",
    "PURPOSIVE": "the claim lives in a to-... / in-order-to-... purpose clause",
    "ATTITUDE": "propositional attitude (the Army views ... as, JP 2-0 recognizes)",
    "CARDINALITY": "numeric counts",
    "REFERENCE": "unresolved anaphora (it, they, this) in the claim",
    "META": "document self-reference or bibliography (ATP 2-22.7 provides ..., see chapter 5, Figure 1-2)",
}

# record_id -> (verdict, blockers, closest supported pattern, note)
TRIAGE: dict[str, tuple[str, tuple[str, ...], str, str]] = {
    "fm2-0-2023-p18-1-3": ("blocked", ("PURPOSIVE", "EVALUATIVE"), "", "A goal statement; effective and flexible carry the content."),
    "fm2-0-2023-p18-1-4": ("blocked", ("MODALITY", "REFERENCE"), "", "Two obligations plus an unresolved they."),
    "fm2-0-2023-p18-1-5": ("partial", ("PURPOSIVE", "REFERENCE"), "subclass_assert", "Only the appositive definition of situational understanding survives."),
    "fm2-0-2023-p19-1-6": ("blocked", ("EVALUATIVE", "REFERENCE", "SCOPED_NEGATION"), "", "Art/science metaphor; the negation sits inside the metaphor."),
    "fm2-0-2023-p19-1-7": ("blocked", ("CARDINALITY", "QUANTIFIER"), "", "one of eight categories needs counts, and generally is a hedge."),
    "fm2-0-2023-p19-1-8": ("blocked", ("EVALUATIVE",), "", "No formal content."),
    "fm2-0-2023-p21-1-11": ("blocked", ("EVALUATIVE",), "", "are crucial for is not a formalizable relation."),
    "fm2-0-2023-p21-1-12": ("blocked", ("EVALUATIVE", "SCOPED_NEGATION"), "", "is key to, plus the Army cannot produce inside a relative clause."),
    "fm2-0-2023-p21-1-13": ("blocked", ("MODALITY", "QUANTIFIER"), "", "Not all echelons, plus should understand."),
    "fm2-0-2023-p21-1-14": ("partial", ("PURPOSIVE",), "binary_assert", "leverages(Analyst, IntelligenceOrganization) drops the entire purpose clause."),
    "fm2-0-2023-p21-1-16": ("partial", ("META",), "binary_assert", "The purview relation survives; the naming convention does not."),
    "fm2-0-2023-p21-1-17": ("blocked", ("EVALUATIVE", "TEMPORAL"), "", "increasingly important as ... is a causal trend claim."),
    "fm2-0-2023-p22-1-18": ("blocked", ("EVALUATIVE", "QUANTIFIER"), "", "unique ... for several reasons, some of which include."),
    "fm2-0-2023-p22-1-19": ("partial", ("ATTRIBUTE", "PURPOSIVE"), "binary_assert", "requires(Commander, Intelligence) drops accurate/relevant/predictive."),
    "fm2-0-2023-p22-1-20": ("expressible", (), "binary_assert", "Two binary assertions: drives and enables. Inseparable is a gloss."),
    "fm2-0-2023-p23-1-21": ("expressible", (), "subclass_assert", "Intelligence subclass-of WarfightingFunction."),
    "fm2-0-2023-p23-1-22": ("blocked", ("MODALITY", "ATTRIBUTE"), "", "must be effective and flexible is the whole claim."),
    "fm2-0-2023-p23-1-23": ("partial", (), "binary_assert", "supports(IntelligenceWarfightingFunction, MilitaryOperation); the IWFT naming is lost."),
    "fm2-0-2023-p23-1-24": ("blocked", ("QUANTIFIER", "TEMPORAL"), "", "often conducted simultaneously."),
    "fm2-0-2023-p24-1-25": ("partial", ("PURPOSIVE",), "binary_assert", "uses(ArmyUnit, IntelligenceProcess); the purpose chain is dropped."),
    "fm2-0-2023-p24-1-26": ("partial", (), "binary_assert", "Needs a providesBasisFor-style relation in the vocabulary."),
    "fm2-0-2023-p24-1-27": ("blocked", ("QUANTIFIER",), "", "Similarity and difference comparatives."),
    "fm2-0-2023-p24-1-28": ("blocked", ("ATTITUDE",), "", "The Army views ... as a model."),
    "fm2-0-2023-p25-1-30": ("partial", ("TEMPORAL",), "nary_assert", "supports(IntelligenceProcess, OperationsProcess) drops performed continuously."),
    "fm2-0-2023-p25-1-31": ("blocked", ("META",), "", "Figure reference."),
    "fm2-0-2023-p26-1-32": ("partial", ("QUANTIFIER", "TEMPORAL"), "subclass_assert", "most basic actions from ... to ... is a superlative over a span."),
    "fm2-0-2023-p26-1-33": ("blocked", ("TEMPORAL",), "", "occurs far in advance of is temporal ordering."),
    "fm2-0-2023-p26-1-34": ("partial", (), "binary_assert", "includes(PlanAndDirectStep, Activity) is near-contentless."),
    "fm2-0-2023-p27-1-35": ("partial", ("PURPOSIVE",), "binary_assert", "synchronizes(Staff, Collection); led by and the purpose clause are lost."),
    "fm2-0-2023-p27-1-36": ("blocked", ("DISJUNCTION", "TEMPORAL"), "", "Four alternative triggers joined by or; condition slots are conjunctive only."),
    "fm2-0-2023-p27-1-37": ("blocked", ("EVALUATIVE",), "", "is difficult."),
    "fm2-0-2023-p27-1-38": ("expressible", (), "subclass_assert", "Definitional: Production subclass-of Development."),
    "fm2-0-2023-p27-1-39": ("blocked", ("MODALITY", "ATTRIBUTE"), "", "must be timely, relevant, accurate, predictive - obligation over attributes."),
    "fm2-0-2023-p27-1-40": ("partial", (), "binary_assert", "A part-of claim over gerunds; term coining required."),
    "fm2-0-2023-p28-1-42": ("blocked", ("MODALITY",), "", "must establish and support."),
    "fm2-0-2023-p28-1-43": ("partial", (), "binary_assert", "add details and areas of emphasis has little formal content."),
    "fm2-0-2023-p28-1-45": ("partial", ("QUANTIFIER",), "nary_assert", "broader perspective than is comparative; the includes-list survives."),
    "fm2-0-2023-p28-1-47": ("partial", ("CONDITION",), "subclass_assert", "The definition survives; when appropriate does not."),
    "fm2-0-2023-p28-1-48": ("partial", ("QUANTIFIER",), "some", "some ?x is-a Analysis is grammatical but says almost nothing."),
    "fm2-0-2023-p29-1-50": ("blocked", ("MODALITY",), "", "must understand and effectively deal with."),
    "fm2-0-2023-p29-1-51": ("expressible", (), "binary_assert", "A clean part-of claim."),
    "fm2-0-2023-p29-1-52": ("partial", ("TEMPORAL",), "binary_assert", "produces(IntelligenceStaff, Assessment) drops continuously and the basis."),
    "fm2-0-2023-p29-1-53": ("partial", ("TEMPORAL",), "binary_assert", "assesses(IntelligenceStaff, InformationCollection) drops continuously."),
    "fm2-0-2023-p30-1-54": ("expressible", (), "binary_assert", "executes(IntelligenceWarfightingFunction, IntelligenceProcess)."),
    "fm2-0-2023-p30-1-55": ("partial", ("PURPOSIVE",), "binary_assert", "The collaboration survives; the appositive definition does not."),
    "fm2-0-2023-p30-1-56": ("partial", ("QUANTIFIER",), "binary_assert", "Dropping primarily strengthens the claim beyond the source."),
    "fm2-0-2023-p30-1-57": ("expressible", (), "nary_assert", "An explicit comprises-list of four named tasks."),
    "fm2-0-2023-p30-1-58": ("blocked", ("PURPOSIVE", "TEMPORAL"), "", "Two chained process clauses with method adjuncts."),
    "fm2-0-2023-p30-1-59": ("partial", ("PURPOSIVE",), "binary_assert", "is used to develop, with a three-way purpose list."),
    "fm2-0-2023-p31-1-61": ("blocked", ("TEMPORAL",), "", "is continuous and occurs throughout."),
    "fm2-0-2023-p31-1-62": ("partial", ("CONDITION", "MODALITY"), "nary_assert", "The includes-list survives; when available and can differ do not."),
    "fm2-0-2023-p32-1-64": ("blocked", ("QUANTIFIER",), "", "Most intelligence collection - no proportional quantifier."),
    "fm2-0-2023-p32-1-65": ("partial", ("PURPOSIVE",), "binary_assert", "The integration survives; the two-step purpose chain does not."),
    "fm2-0-2023-p32-1-66": ("blocked", ("DISJUNCTION", "PURPOSIVE"), "", "detection, identification, analysis, neutralization, or exploitation."),
    "fm2-0-2023-p33-1-67": ("expressible", (), "nary_assert", "consists-of list. This X-capabilities-consist-of frame recurs five times."),
    "fm2-0-2023-p33-1-68": ("expressible", ("META",), "nary_assert", "Grammatical but bibliographic; low query value."),
    "fm2-0-2023-p33-1-70": ("expressible", (), "nary_assert", "consists-of list."),
    "fm2-0-2023-p34-1-71": ("expressible", ("META",), "binary_assert", "Bibliographic."),
    "fm2-0-2023-p34-1-73": ("expressible", (), "nary_assert", "consists-of list."),
    "fm2-0-2023-p34-1-74": ("expressible", ("META",), "nary_assert", "Bibliographic."),
    "fm2-0-2023-p34-1-75": ("expressible", (), "subclass_assert", "Definitional. Check the genus: the source says information, not discipline."),
    "fm2-0-2023-p34-1-77": ("expressible", (), "nary_assert", "consists-of list."),
    "fm2-0-2023-p34-1-78": ("expressible", ("META",), "binary_assert", "Bibliographic."),
    "fm2-0-2023-p35-1-79": ("expressible", (), "subclass_assert", "Definitional. Source genus is intelligence; the rules emit IntelligenceDiscipline - verify."),
    "fm2-0-2023-p35-1-80": ("blocked", ("EVALUATIVE", "QUANTIFIER"), "", "rely heavily on a commander's understanding."),
    "fm2-0-2023-p35-1-82": ("blocked", ("MODALITY",), "", "may obtain is permission, and is the entire content of the sentence."),
    "fm2-0-2023-p35-1-83": ("expressible", ("META",), "nary_assert", "Bibliographic."),
    "fm2-0-2023-p36-1-86": ("expressible", (), "nary_assert", "consists-of list."),
    "fm2-0-2023-p36-1-87": ("expressible", ("META",), "nary_assert", "Bibliographic."),
    "fm2-0-2023-p36-1-88": ("expressible", (), "subclass_assert", "Definitional. Verify the genus before accepting IntelligenceDiscipline."),
    "fm2-0-2023-p36-1-90": ("expressible", (), "nary_assert", "subclass plus an includes-list of four named capabilities."),
    "fm2-0-2023-p36-1-91": ("expressible", ("META",), "binary_assert", "Bibliographic."),
    "fm2-0-2023-p37-1-92": ("partial", ("ATTITUDE", "META"), "nary_assert", "The list survives only if attribution to JP 2-0 may be dropped."),
    "fm2-0-2023-p37-1-93": ("expressible", (), "subclass_assert", "Biometrics subclass-of Process."),
    "fm2-0-2023-p37-1-94": ("partial", ("MODALITY",), "binary_assert", "individuals who may pose a threat - the modal sits in a relative clause."),
    "fm2-0-2023-p37-1-95": ("blocked", ("EVALUATIVE",), "", "a highly successful technique for."),
    "fm2-0-2023-p37-1-96": ("blocked", ("DISJUNCTION",), "", "a device or computer program, and software, firmware, or hardware - twice disjunctive."),
    "fm2-0-2023-p37-1-97": ("blocked", ("EVALUATIVE",), "", "Due to its very nature ... require significant lead time."),
    "fm2-0-2023-p38-1-98": ("partial", ("SCOPED_NEGATION",), "subclass_assert", "Dropping are not publicly available erases what distinguishes DOMEX from OSINT."),
    "fm2-0-2023-p38-1-99": ("partial", ("ATTRIBUTE",), "subclass_assert", "DOMEX subclass-of Mission; the requirements list is adjectival."),
    "fm2-0-2023-p38-1-100": ("partial", ("DISJUNCTION", "PURPOSIVE"), "subclass_assert", "The genus survives; to control ... or to attack does not."),
    "fm2-0-2023-p38-1-102": ("partial", ("DISJUNCTION",), "subclass_assert", "The genus survives; locate or localize does not."),
    "fm2-0-2023-p38-1-103": ("partial", ("DISJUNCTION",), "subclass_assert", "The genus survives; degrade, neutralize, or destroy does not."),
    "fm2-0-2023-p39-1-104": ("expressible", (), "subclass_assert", "ForensicScience subclass-of Process."),
    "fm2-0-2023-p39-1-105": ("blocked", ("MODALITY",), "", "can assist in is the whole claim."),
    "fm2-0-2023-p39-1-106": ("expressible", (), "subclass_assert", "Definitional. Verify the genus before accepting IntelligenceProduct."),
    "fm2-0-2023-p39-1-108": ("partial", ("PURPOSIVE",), "nary_assert", "A five-way purpose list plus a cross-reference."),
    "fm2-0-2023-p40-1-109": ("blocked", ("TEMPORAL", "QUANTIFIER"), "", "currently, plus echelons corps and above."),
    "fm2-0-2023-p41-1-111": ("expressible", ("META",), "nary_assert", "Bibliographic."),
    "fm2-0-2023-p41-1-112": ("expressible", (), "nary_assert", "A consists-of list over four named capability kinds."),
    "fm2-0-2023-p41-1-113": ("partial", ("CONDITION",), "binary_assert", "when planning information collection operations is a gerund condition."),
    "fm2-0-2023-p41-1-114": ("blocked", ("MODALITY", "CONDITION"), "", "are responsible for, under a gerund condition."),
    "fm2-0-2023-p42-1-115": ("expressible", (), "nary_assert", "A compilation-of list over five named components."),
    "fm2-0-2023-p42-1-117": ("partial", ("ATTRIBUTE",), "binary_assert", "requires(IntelligenceArchitecture, IntelligenceProfessional) drops the qualifications."),
    "fm2-0-2023-p43-1-118": ("blocked", ("MODALITY", "META"), "", "can be exemplified through."),
    "fm2-0-2023-p43-1-119": ("blocked", ("META",), "", "A see-also imperative."),
    "fm2-0-2023-p43-1-120": ("blocked", ("DISJUNCTION", "EVALUATIVE"), "", "whether a peer threat or terrorist cell, plus significant challenge."),
    "fm2-0-2023-p43-1-121": ("blocked", ("TEMPORAL", "EVALUATIVE"), "", "constantly changes."),
    "fm2-0-2023-p44-1-123": ("blocked", ("MODALITY", "QUANTIFIER"), "", "most challenging, plus the threat can apply."),
    "fm2-0-2023-p44-1-125": ("partial", ("ATTRIBUTE",), "binary_assert", "requires(InformationCollection, Planning) drops thorough/creative/aggressive."),
}

VERDICTS = ("expressible", "partial", "blocked")


def build(candidates: list[dict]) -> list[dict]:
    by_id = {r["record_id"]: r for r in candidates}
    if len(by_id) != len(candidates):
        raise SystemExit("duplicate record_id in candidates")
    missing = sorted(set(by_id) - set(TRIAGE))
    unknown = sorted(set(TRIAGE) - set(by_id))
    if missing:
        raise SystemExit(f"no triage verdict for {len(missing)} candidate(s): {missing[:5]}")
    if unknown:
        raise SystemExit(f"triage references {len(unknown)} unknown record(s): {unknown[:5]}")
    rows = []
    for record_id, candidate in by_id.items():
        verdict, blockers, pattern, note = TRIAGE[record_id]
        if verdict not in VERDICTS:
            raise SystemExit(f"{record_id}: bad verdict {verdict!r}")
        for code in blockers:
            if code not in BLOCKERS:
                raise SystemExit(f"{record_id}: unknown blocker {code!r}")
        rows.append({
            "record_id": record_id,
            "pdf_page": candidate["pdf_page"],
            "paragraph": candidate["paragraph"],
            "nl": candidate["nl"],
            "surface_tags": ",".join(candidate.get("phenomena", [])),
            "ceiling_verdict": verdict,
            "blockers": ",".join(blockers),
            "closest_pattern": pattern,
            "loss_note": note,
            "source": "claude-draft-triage",
            "status": "draft",
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    verdicts = Counter(r["ceiling_verdict"] for r in rows)
    blockers = Counter(c for r in rows for c in r["blockers"].split(",") if c)
    bibliographic = sum(
        1 for r in rows if r["ceiling_verdict"] == "expressible" and "META" in r["blockers"]
    )
    blocked_only = Counter(
        c
        for r in rows
        if r["ceiling_verdict"] == "blocked"
        for c in r["blockers"].split(",")
        if c
    )
    return {
        "total": len(rows),
        "verdicts": {v: verdicts[v] for v in VERDICTS},
        "expressible_bibliographic": bibliographic,
        "expressible_substantive": verdicts["expressible"] - bibliographic,
        "blockers_all": dict(blockers.most_common()),
        "blockers_on_blocked_only": dict(blocked_only.most_common()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draft CNL expressibility triage")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)

    candidates = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = build(candidates)
    summary = summarize(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(
        json.dumps({"summary": summary, "blocker_glossary": BLOCKERS}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
