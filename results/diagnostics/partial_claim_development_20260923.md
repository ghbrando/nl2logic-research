# FM 2-0 partial-claim development review — 23 September 2026

The [expanded development packet](../../data/benchmarks/fm2-0/development_partial_claims_20260923.json) reviews 14 deliberately selected passages from FM 2-0 (01 October 2023), chapter 1. Three have an AI-reviewed, source-entailed **partial** subclass claim; eleven are explicit abstentions. This is a diagnostic sample, not a representative estimate of passage-level coverage. No model was queried and no training or GPU job was run.

| Decision | Paragraphs | Finding |
| --- | --- | --- |
| Partial positive | 1-79, 1-88, 1-93 | Produced open-source and technical intelligence can be mapped to `IntelligenceProduct` in their product senses; biometrics can be mapped to `Process` in its process sense. New class symbols are declared separately from the source assertions. |
| Preloaded axiom | 1-21, 1-98 | The apparent positive is already asserted or closely entailed by `doctrine_domain.kif`; it cannot measure source formalization. |
| Scope or qualifier | 1-8, 1-62 | A doctrinal scope or availability/planning condition would be lost. |
| Cardinality | 1-26 | The four steps and five activities cannot be stated faithfully in current CNL. |
| Relation semantics | 1-51, 1-100, 1-102 | *Part* and *division* cannot safely be guessed to mean subclass or another available relation. |
| Comparison | 1-59 | Relative reliability, time, and deception susceptibility lack a faithful pattern. |
| Disjunction | 1-96 | A device-or-program definition cannot be converted to either universal subclass alone. |
| Representation gap | 1-104 | “Application of scientific processes” does not justify a simple class mapping for forensic science. |

All positives are narrow taxonomy claims. The packet does **not** demonstrate reliable relation, condition, comparison, cardinality, or full-sentence formalization. The two new intelligence-product classes distinguish the produced-intelligence sense from the already-loaded `OpenSourceIntelligence` and `TechnicalIntelligence` discipline classes. That sense distinction and the omitted parts of each definition are recorded in the packet. AI review and the mechanical validator do not replace a domain expert or prove model performance.

The [validator](../../scripts/validate_claim_packet.py) now requires a coded rationale for abstentions and rejects an abstention that carries a proposed formula. It checks source provenance and the syntax of positive claims but cannot decide entailment. No conclusion about accuracy should be drawn from these 14 development passages.

A separate [chapter 2 evaluation source selection](../../data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/README.md) now pins all 29 eligible paragraphs on PDF pages 45–52, with no labels or model outputs. Its manifest fingerprints the selected excerpts, page extracts, ontology inputs, and claim contract. Exact first-sentence overlap with the existing FM 2-0 seed and chapter 1 candidate files was zero. During extraction, chapter 2 footers exposed a page-number parser bug: `FM 2-0` was sometimes mistaken for printed page `2-0`. The parser and frozen source rows now use the actual printed page numbers `2-1` through `2-8`, with regression tests for both footer orders.

The next review step is to annotate these 29 selected passages independently, recording defensible partial claims or explicit abstentions with exact source spans. Freeze those AI-only labels before a downstream model query. The source batch alone is **not** an evaluation gold set; training remains on hold.

Validation: 986 tests passed, one skipped. No training or GPU job was started.
