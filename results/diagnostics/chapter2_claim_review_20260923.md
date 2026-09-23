# FM 2-0 chapter 2 partial-claim review — 23 September 2026

The [AI-reviewed packet](../../data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/ai_reviewed_claims_20260923.json) labels all 29 previously selected paragraphs on PDF pages 45–52, before any target NL2Logic model query. The [review manifest](../../data/benchmarks/fm2-0/chapter2-2023-claim-eval-source/review_manifest.json) pins the label and source-selection hashes. One paragraph has a defensible **partial** positive; 28 have explicit abstentions. No training or GPU job was run.

The positive is paragraph 2-26, printed page 2-8. The source explicitly calls “consolidate gains” activities. A new class symbol is declared separately, and `ConsolidatingGains subclass-of Process` captures only that these activities are processes. It omits the purpose of making success enduring, setting sustainable conditions, and transferring control. This narrow assertion is not preloaded in `doctrine_domain.kif`. It is AI-reviewed and provisional, not expert gold.

| Abstention reason | Passages |
| --- | ---: |
| Causality | 5 |
| Scope or qualifier | 4 |
| Representation gap | 4 |
| Temporal context | 4 |
| Modality or quantifier | 3 |
| Named entity | 2 |
| Disjunction | 2 |
| Nonassertive reference | 1 |
| Incomplete source at page break | 1 |
| Cardinality | 1 |
| Relation semantics | 1 |

The result reinforces the current bottleneck: the closed CNL and vocabulary can express a few narrow taxonomy claims but little of the conditions, quantities, causal structure, and actor-specific doctrine in these passages. **One positive is too few for a stable positive-recall estimate.** The packet can support a later abstention/false-accept diagnostic, but should not be presented as a balanced accuracy benchmark. The mechanical validator checks provenance, full row coverage, frozen ontology inputs, declarations, CNL, and compiled KIF; it cannot prove entailment. AI-only labels may still contain errors, so any later model comparison must report that limitation and keep this packet separate from training.

The frozen text hashes use Git-canonical LF line endings, so the same label and source manifests verify on Windows and Linux. Validation: 992 tests passed, one skipped. The target NL2Logic model was not queried.
