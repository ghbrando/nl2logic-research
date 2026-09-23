# Prior-logic memory probe — 23 September 2026

An opt-in research path now retrieves relevant existing logic statements and puts them in the seq2seq encoder prompt. The decoder can attend to that text while generating tokens. The production source gate and output grammar still receive only the current source phrase; retrieved logic cannot make an unsupported input eligible or add a class to the closed vocabulary. All proposed outputs still require strict compilation, grounding, and semantic review. No training was performed.

There is no committed corpus of accepted, source-backed formal claims yet. This first mechanism test therefore uses the 40 subclass axioms in the curated `doctrine_domain.kif` as **background memory**, never as evidence for a new claim. Retrieval uses literal overlap with the child class name, keeps one statement per candidate, and records its file hash and line number. The [paired-run comparison](partial_claim_memory_cpu_20260923_comparison.json) verifies that both arms used the same 14 unlabeled chapter 1 development sources, checkpoint, code, extraction results, and source-only grammar.

| Development result | Baseline | Retrieved-memory arm |
| --- | ---: | ---: |
| Extracted candidates | 3 | 3 |
| Candidates with retrieved memory | 0 | 2 |
| Stopped before generation by source grammar | 2 | 2 |
| Invalid unbound-variable output | 1 | 1 |
| Accepted claims | 0 | 0 |

The two retrieved axioms were `(subclass OpenSourceIntelligence IntelligenceDiscipline)` and `(subclass TechnicalIntelligence IntelligenceDiscipline)`, with line provenance in the [memory predictions](partial_claim_memory_cpu_20260923_retrieved/predictions.jsonl). Those sources are the same intelligence passages whose provisional reviewed claims concern the **product** sense. The existing discipline axioms must not be copied as answers to them. Both passages lacked a lexically supported constrained continuation from their source phrase, so generation stopped **before** the model could use memory. The biometrics passage retrieved no specific axiom and again generated `?x is-a Process`, which strict compilation rejects because `?x` is unbound. The [baseline](partial_claim_memory_cpu_20260923_baseline/scoring/summary.json) and [memory](partial_claim_memory_cpu_20260923_retrieved/scoring/summary.json) scores are identical. This development run cannot estimate a benefit from attention to old statements.

A separate [engineering control](partial_claim_memory_cpu_20260923_control/control.json) used a source sentence that restates a preloaded ontology axiom. Both arms reached model generation, and their encoder-prompt hashes differ, confirming the memory text was supplied to the actual model path. Both generated the same invalid `?x is-a OpenSourceIntelligence` with an unbound variable. The control is an ontology restatement, not an independent accuracy example.

The next scientific prerequisite is a small, reviewed memory of **accepted source-backed claims** with exact provenance and explicit sense, plus a development set whose source-only grammar admits generation. Only then can a paired test ask whether retrieved prior claims help a *new* source claim. A separate, reviewed declaration step is needed for the missing product/process sense classes; memory must not create those classes on its own. Use fresh unseen passages for any later independent evaluation because the current development and chapter 2 passages have already been examined. This run was CPU-only in the project-owned rootless container on worker `192.168.94.13`; the existing GPU inference service was left in place.
