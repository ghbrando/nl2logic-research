# nl2logic-research

Research code for a precision-first `NL -> CNL -> KIF` pipeline for military doctrine formalization.

The active architecture is centered on constrained seq2seq decoding, CNL compilation, and grounding-based acceptance. See [ARCHITECTURE.md](ARCHITECTURE.md) for the problem statement, solution proposal, primary pipeline, and archived baseline notes.

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
