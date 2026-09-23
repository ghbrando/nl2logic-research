"""CPU smoke control for the memory encoder path; not an evaluation example."""
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

from src.compiler.compiler import CNLCompiler
from src.eval.model_predictor import load_model_and_tokenizer
from src.fsm.cnl_fsm import CNLSampler
from src.reasoning.claim_memory import load_ontology_memory, render_memory_prompt, retrieve_memory
from src.training.train import format_prompt

CONTROL_SOURCE = "OpenSourceIntelligence is a type of IntelligenceDiscipline."
CONTROL_AXIOM = "(subclass OpenSourceIntelligence IntelligenceDiscipline)"


def run(model_path: Path, output_dir: Path) -> None:
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Control requires a CPU-only container")
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(model_path)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Control model is not on CPU")
    sampler = CNLSampler(model, tokenizer, allow_background_axioms=False)
    memory = retrieve_memory(CONTROL_SOURCE, load_ontology_memory(), limit=1)
    if len(memory) != 1 or memory[0].kif != CONTROL_AXIOM:
        raise ValueError("Control memory axiom is missing or changed")
    source_prompt = format_prompt(CONTROL_SOURCE)
    compiler = CNLCompiler()
    arms = []
    for mode, statements in (("none", []), ("retrieved", memory)):
        model_prompt = render_memory_prompt(source_prompt, statements)
        cnl, kif, error = None, None, None
        try:
            cnl = sampler.sample(source_prompt, model_prompt=model_prompt, max_tokens=64)
            kif = compiler.compile(cnl, require_closed=True, require_argument_kinds=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        arms.append({
            "mode": mode,
            "memory_statement_ids": [statement.statement_id for statement in statements],
            "model_prompt_sha256": hashlib.sha256(model_prompt.encode("utf-8")).hexdigest(),
            "cnl": cnl,
            "compiled_kif": kif,
            "error": error,
        })
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "purpose": "Engineering smoke control only: source restates a preloaded ontology axiom and is not independent claim evidence.",
        "source": CONTROL_SOURCE,
        "background_axiom": CONTROL_AXIOM,
        "source_prompt_sha256": hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
        "adapter_sha256": hashlib.sha256((model_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "device": "cpu",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "arms": arms,
    }
    (output_dir / "control.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
