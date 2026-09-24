"""CPU-only oracle-declaration and prior-logic development ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_claim_predictions import text_sha256
from scripts.run_partial_claim_dev_probe import load_sources
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler, UnsupportedInputError
from src.ingest.grounding import assess_declared_definition
from src.ontology.vocab import load_closed_class_terms
from src.preprocessing.declarations import (
    PROPOSED_STATUS, load_declaration_registry, match_declaration, parent_candidates,
)
from src.preprocessing.partial_claims import extract_paragraph_candidate
from src.reasoning.claim_memory import (
    load_ontology_memory, load_prior_claim_memory, render_memory_prompt, retrieve_memory,
    retrieve_prior_by_genus,
)
from src.training.train import format_prompt


def rules_definition(passage: str, evidence: str, declaration: dict, parents: list[str]) -> str | None:
    """Rules-only arm: emit the one parent the current passage states as genus."""
    accepted = [
        cnl for cnl in (f"{declaration['symbol']} subclass-of {parent}" for parent in parents)
        if assess_declared_definition(cnl, passage=passage, evidence_sentence=evidence,
                                      declaration=declaration, parent_terms=parents).accepted
    ]
    return accepted[0] if len(accepted) == 1 else None


def gate(cnl: str | None, passage: str, evidence: str, declaration: dict, parents: list[str]) -> dict:
    if not cnl:
        return {"accepted": False, "reason": "no_output"}
    result = assess_declared_definition(cnl, passage=passage, evidence_sentence=evidence,
                                        declaration=declaration, parent_terms=parents)
    return {"accepted": result.accepted, "reason": result.reason}


def declaration_parents(registry: dict, declaration: dict) -> list[str]:
    pool = list(registry["existing_parent_pool"])
    if registry["status"] != PROPOSED_STATUS:
        return pool
    return parent_candidates(declaration, pool, load_closed_class_terms())


def predict(source_path: Path, source_manifest_path: Path, registry_path: Path,
            model_path: Path, output_dir: Path, *, max_new_tokens: int = 64,
            prior_claims_paths: list[Path] | None = None) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rows = load_sources(source_path, source_manifest_path)
    registry = load_declaration_registry(registry_path, source_packet_sha256=source_manifest.get("packet_sha256"),
                                         sources_sha256=source_manifest["sources_sha256"])
    prepared = []
    for row in rows:
        candidate, screened = extract_paragraph_candidate(row["source_excerpt"], row["source_sentence"])
        evidence = row["source_excerpt"][screened[-1]["start"]:screened[-1]["end"]] if candidate.status == "candidate" else None
        declaration = match_declaration(candidate.candidate_text, evidence, registry) if evidence else None
        prepared.append((row, candidate, screened, evidence, declaration))
    eligible = [item for item in prepared if item[4] is not None and not item[1].gate_reasons]

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Declaration probe unexpectedly has CUDA access")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path) if eligible else (None, None)
    if model is not None and any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Declaration probe requires a CPU-only model")
    if model is not None:
        model.eval()
    memory_statements = load_ontology_memory()
    prior_claims = [statement for path in prior_claims_paths or [] for statement in load_prior_claim_memory(path)]
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "protocol": "Development-only oracle class declarations, no subclass assertions; full current evidence sentence plus source head in encoder; source-head gate; decoder fixes the declared term as child and offers only the existing parent pool; rules-only, no-memory, and ontology-memory arms each reviewed by the passage-only declared-definition gate. Labels read only after predictions are saved.",
        "decoder_constraint": "definition_child",
        "gate": "assess_declared_definition",
        "arms": ["rules", "baseline", "retrieved"],
        "sources_sha256": text_sha256(source_path),
        "source_manifest_sha256": text_sha256(source_manifest_path),
        "registry_sha256": text_sha256(registry_path),
        "memory_source_sha256": memory_statements[0].source_sha256,
        "prior_claims_paths": [path.as_posix() for path in prior_claims_paths or []],
        "prior_claims_sha256": sorted({statement.source_sha256 for statement in prior_claims}),
        "prior_claim_count": len(prior_claims),
        "adapter_sha256": hashlib.sha256((model_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "device": "cpu",
        "generation": {"decoder": "CNLSampler", "num_beams": 4, "max_new_tokens": max_new_tokens},
        "source_count": len(rows),
        "eligible_count": len(eligible),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row, candidate, screened, evidence, declaration in prepared:
            arms = {}
            prior = retrieve_memory(candidate.candidate_text, prior_claims, limit=1) if declaration else []
            if declaration and not prior and declaration.get("genus_phrase"):
                prior = retrieve_prior_by_genus(declaration["genus_phrase"], prior_claims, limit=1)
            memory = (retrieve_memory(candidate.candidate_text, memory_statements, limit=1) + prior) if declaration else []
            parents = declaration_parents(registry, declaration) if declaration else []
            if declaration is not None and not candidate.gate_reasons:
                allowed = set(parents) | {declaration["symbol"]}
                sampler = CNLSampler(model, tokenizer, allow_background_axioms=False,
                                     declared_class_terms=allowed, allow_definition_subclass=True,
                                     definition_child=declaration["symbol"])
                source = row["source_excerpt"]
                rules_cnl = rules_definition(source, evidence, declaration, parents)
                arms["rules"] = {"cnl": rules_cnl, "decoder_abstention": None, "error": None,
                                 "gate": gate(rules_cnl, source, evidence, declaration, parents)}
                source_prompt = format_prompt(candidate.candidate_text)
                base_model_prompt = (
                    f"{source_prompt}\nCurrent source evidence: {evidence}\n"
                    f"Approved class name only: {declaration['declaration']}\n"
                    f"Available existing classes: {', '.join(parents)}.\n"
                    "Produce a new claim only if the current source supports it."
                )
                for mode, statements in (("baseline", []), ("retrieved", memory)):
                    model_prompt = render_memory_prompt(base_model_prompt, statements)
                    cnl, abstention, error = None, None, None
                    try:
                        cnl = sampler.sample(source_prompt, model_prompt=model_prompt,
                                             max_tokens=max_new_tokens)
                    except UnsupportedInputError as exc:
                        abstention = str(exc)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    arms[mode] = {
                        "model_prompt_sha256": hashlib.sha256(model_prompt.encode("utf-8")).hexdigest(),
                        "cnl": cnl,
                        "decoder_abstention": abstention,
                        "error": error,
                        "gate": gate(cnl, source, evidence, declaration, parents),
                    }
            result = {
                "record_id": row["record_id"],
                "paragraph": row["paragraph"],
                "source_sha256": hashlib.sha256(row["source_excerpt"].encode("utf-8")).hexdigest(),
                "extraction": asdict(candidate),
                "screened_sentences": screened,
                "evidence_sentence": evidence,
                "declaration": declaration,
                "approved_class_terms": sorted(set(parents) | {declaration["symbol"]}) if declaration else [],
                "parent_candidates": parents,
                "memory": [asdict(statement) for statement in memory],
                "arms": arms,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{row['paragraph']} declaration={declaration['symbol'] if declaration else None} " + " ".join(f"{mode}={arms[mode]['cnl']}:{arms[mode]['gate']}" for mode in arms), flush=True)
    manifest["prediction_count"] = len(rows)
    manifest["predictions_sha256"] = text_sha256(prediction_path)
    manifest["status"] = "completed"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prior-claims", type=Path, nargs="*", default=None,
                        help="scored.jsonl files of separate runs; their gate-accepted rules claims become memory")
    args = parser.parse_args()
    predict(args.sources, args.source_manifest, args.registry, args.model_path,
            args.output_dir, max_new_tokens=args.max_new_tokens, prior_claims_paths=args.prior_claims)


if __name__ == "__main__":
    main()
