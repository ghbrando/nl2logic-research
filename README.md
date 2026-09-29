# nl2logic-research

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/T5-constrained_decoding-0b0b0b?style=flat-square&logo=huggingface&logoColor=white" alt="T5-constrained decoding" />
  <img src="https://img.shields.io/badge/Vampire-theorem_prover-6A1B9A?style=flat-square" alt="Vampire-theorem prover" />
  <img src="https://img.shields.io/badge/KIF%2FSUMO-555555?style=flat-square" alt="KIF%2FSUMO" />
</p>

<p align="center"><img src="docs/demo/demo.gif" width="860" alt="Terminal demo: a doctrine query comes back PROVEN with its cited source passage, and a second query comes back UNKNOWN" /></p>

Research code for a precision-first `NL -> CNL -> KIF` pipeline for military doctrine formalization.

The active architecture is centered on constrained seq2seq decoding, CNL compilation, and grounding-based acceptance. See [ARCHITECTURE.md](ARCHITECTURE.md) for the problem statement, solution proposal, primary pipeline, and archived baseline notes.

## At a glance

**Goal:** turn authoritative military doctrine into a prover-ready KIF/SUMO knowledge base. Every accepted statement must trace back to its source text, and the system should abstain rather than hallucinate.

```text
doctrine PDF ─► sentence extraction ─► decomposition + term linking
            ─► fine-tuned T5 → CNL, decoded under a grammar FSM (invalid tokens pruned at each step)
            ─► CNL → KIF compiler ─► grounding gate ─► accept / review / reject
            ─► domain theory ─► theorem prover (Vampire) ─► proof-backed answers
```

The core claim: **constraining generation at decode time** raises formalization precision compared to free-form LLM output. The work is organized as a 5-paper series in [`papers/`](papers): architecture, decidability, benchmark, commonsense, and generalization.

## Proof-backed classification demo

The [reasoning example](examples/reasoning/README.md) translates a small ground
classification subset into TPTP, checks consistency with Vampire, and answers
with named source evidence and explicit background assumptions. See the example
for setup, commands, supported semantics, and validation.

## Formalizer comparison

Use the [comparison harness](data/benchmarks/fm2-0/COMPARISON.md) to evaluate
rules-only, raw, and constrained generation with shared gates. A
[100-passage review packet](data/benchmarks/fm2-0/chapter1-2023-review/README.md)
is available for FM 2-0 (2023), Chapter 1. Drafts remain unscored until reviewed.
