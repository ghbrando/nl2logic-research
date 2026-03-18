"""Test structured extraction on real FM 2-0 doctrine sentences.

Usage (local model on DGX Spark — default):
    python scripts/test_structured_extraction.py

    Uses Mistral-7B-Instruct by default.  Override with:
    python scripts/test_structured_extraction.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

Usage (Anthropic API — optional):
    python scripts/test_structured_extraction.py --backend anthropic --api-key $ANTHROPIC_API_KEY

This script:
1. Takes a curated set of real FM 2-0 sentences (mix of formalizable and not)
2. Runs the structured extractor on each
3. Maps extracted claims to SUMO terms
4. Assembles KIF
5. Prints a detailed report comparing extraction quality
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.ingest.structured_extractor import (
    AssembledFact,
    ExtractionResult,
    MappedClaim,
    SUMOMapper,
    StructuredExtractor,
    assemble_kif,
)

# ── Test sentences from FM 2-0 pages 20-24 ─────────────────────────────────
# Mix of formalizable and non-formalizable doctrine text.

_TEST_SENTENCES: list[dict] = [
    # --- Should be formalizable ---
    {
        "sentence": "Combat information is a report that is gathered by or provided to the tactical commander.",
        "page": 3,
        "expected_formalizable": True,
        "notes": "Instance pattern: combat info is-a report",
    },
    {
        "sentence": "The intelligence warfighting function is the related tasks and systems that facilitate understanding the enemy, terrain, weather, civil considerations, and other significant aspects of the operational environment.",
        "page": 4,
        "expected_formalizable": True,
        "notes": "Instance/definitional: IWF is-a function",
    },
    {
        "sentence": "The intelligence warfighting function supports operations through a broad range of supporting doctrinal tasks.",
        "page": 4,
        "expected_formalizable": True,
        "notes": "Relation: IWF supports operations",
    },
    {
        "sentence": "Intelligence is the product resulting from the collection, processing, integration, evaluation, analysis, and interpretation of available information concerning foreign nations, hostile or potentially hostile forces or elements.",
        "page": 1,
        "expected_formalizable": True,
        "notes": "Instance/definitional: intelligence is-a product",
    },
    {
        "sentence": "Army intelligence professionals are the Soldiers and Department of the Army Civilians who perform intelligence tasks and provide intelligence to commanders and staffs.",
        "page": 3,
        "expected_formalizable": True,
        "notes": "Instance/relation: professionals are soldiers who provide intelligence",
    },
    {
        "sentence": "The intelligence enterprise is the collective resources of the intelligence community.",
        "page": 2,
        "expected_formalizable": True,
        "notes": "Instance: enterprise is-a collective/organization",
    },
    {
        "sentence": "Military intelligence is a category of intelligence relating to military capabilities and operations.",
        "page": 1,
        "expected_formalizable": True,
        "notes": "Subclass or instance: MI is a category of intelligence",
    },
    # --- Should NOT be formalizable ---
    {
        "sentence": "It must flow continuously to the commander and staff before, during, and after an operation.",
        "page": 1,
        "expected_formalizable": False,
        "notes": "Unresolved pronoun 'it', procedural",
    },
    {
        "sentence": "However, due to the unique characteristics of Army operations, the Army intelligence process steps differ in some important but subtle ways.",
        "page": 1,
        "expected_formalizable": False,
        "notes": "Vague comparative, no concrete factual assertion",
    },
    {
        "sentence": "See ATP 2-33.4 for more information.",
        "page": 1,
        "expected_formalizable": False,
        "notes": "Citation reference, not a factual claim",
    },
    {
        "sentence": "The most important element is the people who make it work.",
        "page": 2,
        "expected_formalizable": False,
        "notes": "Vague superlative, unresolved 'it'",
    },
    {
        "sentence": "Army intelligence professionals should understand what they receive from the intelligence enterprise and know what it can and cannot provide.",
        "page": 2,
        "expected_formalizable": False,
        "notes": "Procedural guidance with pronouns",
    },
    # --- Borderline cases ---
    {
        "sentence": "Intelligence reduces uncertainty about aspects of the operational environment by identifying the threats, their capabilities, and potential courses of action.",
        "page": 1,
        "expected_formalizable": True,
        "notes": "Relation: intelligence reduces uncertainty (borderline - 'their' pronoun)",
    },
    {
        "sentence": "The intelligence process is a model and common framework to guide Army professionals in their thoughts, discussions, plans, and assessments about intelligence.",
        "page": 5,
        "expected_formalizable": True,
        "notes": "Instance: intelligence process is-a model/framework",
    },
]


def _print_separator():
    print("=" * 80)


def _print_claim_detail(claim, mapped: MappedClaim | None, fact: AssembledFact | None):
    print(f"    Pattern:    {claim.pattern}")
    print(f"    Subject:    '{claim.subject}'")
    print(f"    Predicate:  '{claim.predicate}'")
    print(f"    Object:     '{claim.object}'")
    print(f"    Quantifier: {claim.quantifier} | Negated: {claim.negated}")
    print(f"    Confidence: {claim.confidence}")
    print(f"    Source:     '{claim.source_span}'")

    if mapped:
        print(f"    -> SUMO subject:  {mapped.subject_class}")
        print(f"    -> SUMO relation: {mapped.relation}")
        print(f"    -> SUMO object:   {mapped.object_class}")
    else:
        print("    -> SUMO mapping:  FAILED (no SUMO term match)")

    if fact:
        print(f"    => CNL: {fact.cnl}")
        print(f"    => KIF: {fact.kif}")
    elif mapped:
        print("    => KIF assembly:  FAILED")


def main():
    parser = argparse.ArgumentParser(description="Test structured extraction on FM 2-0 sentences")
    parser.add_argument(
        "--backend", default="local", choices=["local", "anthropic"],
        help="Extraction backend: 'local' (HuggingFace model on GPU) or 'anthropic' (API)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name/path. Defaults: local=Mistral-7B-Instruct-v0.3, anthropic=claude-sonnet",
    )
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"), help="Anthropic API key")
    parser.add_argument("--device", default=None, help="Device for local model (auto-detected)")
    args = parser.parse_args()

    if args.backend == "anthropic" and not args.api_key:
        print("Error: Anthropic backend requires --api-key or ANTHROPIC_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing structured extractor (backend={args.backend})...")
    extractor = StructuredExtractor(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        device=args.device,
    )
    mapper = SUMOMapper()

    total = len(_TEST_SENTENCES)
    correct_formalizability = 0
    total_claims_extracted = 0
    total_claims_mapped = 0
    total_facts_assembled = 0

    all_facts: list[dict] = []

    for i, test_case in enumerate(_TEST_SENTENCES):
        sentence = test_case["sentence"]
        expected = test_case["expected_formalizable"]

        _print_separator()
        print(f"[{i+1}/{total}] Page {test_case['page']}")
        print(f"Sentence: {sentence}")
        print(f"Expected formalizable: {expected}")
        print(f"Notes: {test_case['notes']}")
        print()

        try:
            result = extractor.extract(sentence)
        except Exception as exc:
            print(f"  EXTRACTION ERROR: {exc}")
            continue

        actual = result.formalizable
        match = actual == expected
        correct_formalizability += int(match)
        status = "CORRECT" if match else "MISMATCH"

        print(f"  LLM says formalizable: {actual}  [{status}]")
        if result.reason:
            print(f"  Reason: {result.reason}")
        print(f"  Claims extracted: {len(result.claims)}")

        for j, claim in enumerate(result.claims):
            total_claims_extracted += 1
            print(f"\n  Claim {j+1}:")

            mapped = mapper.map_claim(claim)
            fact = None
            if mapped:
                total_claims_mapped += 1
                fact = assemble_kif(mapped)
                if fact:
                    total_facts_assembled += 1
                    all_facts.append({
                        "sentence": sentence,
                        "page": test_case["page"],
                        "cnl": fact.cnl,
                        "kif": fact.kif,
                        "pattern": fact.pattern,
                        "confidence": fact.confidence,
                        "source_span": fact.source_span,
                    })

            _print_claim_detail(claim, mapped, fact)

        print()

    _print_separator()
    print("SUMMARY")
    _print_separator()
    print(f"Total sentences:            {total}")
    print(f"Formalizability correct:    {correct_formalizability}/{total} ({100*correct_formalizability/total:.0f}%)")
    print(f"Claims extracted:           {total_claims_extracted}")
    print(f"Claims mapped to SUMO:      {total_claims_mapped}")
    print(f"Facts assembled (CNL+KIF):  {total_facts_assembled}")
    print()

    if all_facts:
        print("ASSEMBLED FACTS:")
        _print_separator()
        for fact in all_facts:
            print(f"  KIF: {fact['kif']}")
            print(f"  CNL: {fact['cnl']}")
            print(f"  Src: {fact['source_span']}")
            print(f"  Confidence: {fact['confidence']} | Pattern: {fact['pattern']}")
            print()

    # Write results to JSON for analysis
    output_path = _REPO_ROOT / "data" / "ontology" / "extraction_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total": total,
                    "formalizability_correct": correct_formalizability,
                    "claims_extracted": total_claims_extracted,
                    "claims_mapped": total_claims_mapped,
                    "facts_assembled": total_facts_assembled,
                },
                "facts": all_facts,
            },
            f,
            indent=2,
        )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
