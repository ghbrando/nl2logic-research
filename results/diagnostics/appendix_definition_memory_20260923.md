# Fresh FM 2-0 appendix run after the head-noun revision — 23 September 2026

This run follows the [appendix protocol](appendix_definition_protocol_20260923.md). The rules were fixed at `4c0bebe` before any appendix text was extracted.

- **Sources.** [52 appendix paragraphs](../../data/benchmarks/fm2-0/appendices-definition-unseen-source/manifest.json), selected label-free from 193 scanned. None had been selected or queried before.
- **Declarations.** 25 source-only declarations were applied. 27 passages had no declaration and abstained.
- **Labels.** [52 provisional AI labels](../../data/benchmarks/fm2-0/appendices-definition-unseen-source/ai_reviewed_claims_20260923.json) (2 positive), frozen at `7dc3381` before the query. The reviewer also wrote the gate, so the labels are neither independent nor expert gold.
- **Run.** One CPU job at `7dc3381` in the project-owned container on worker `192.168.94.13`, with no training. The other user's GPU service was untouched. See the [predictions and manifest](appendix_definition_cpu_20260923/manifest.json) and the [scoring](appendix_definition_cpu_20260923/scoring/summary.json).

| Arm | Gate-accepted | Matches label | **Unsupported** | Label positives covered |
| --- | --- | --- | --- | --- |
| Rules only | 3 | 2 | 1 | 2 / 2 |
| Model, no memory | 2 | 2 | **0** | 2 / 2 |
| Model, retrieved memory | 2 | 2 | **0** | 2 / 2 |

- **Accepted by every arm:** `IntelligencePreparationOfTheOperationalEnvironmentProcess subclass-of Process` (B-39) and `SituationDevelopmentProcess subclass-of Process` (B-44).
- **Rules-only error:** the rules arm also accepted `SituationalUnderstandingProduct subclass-of Product` (B-18, "the product of applying analysis"). In SUMO, `Product` means a manufactured artifact, so the word matches but the sense does not. Where the named parent is outside the reviewed genus table, the whole-phrase rule cannot detect a sense mismatch.

## What this does and does not show

- **Unsupported claims.** On this fresh set, both model arms accepted no unsupported claims and covered both label positives. The rules arm covered the same positives but let one sense error through. Compared with the [chapter 3+ run](unseen_definition_memory_20260923.md), where unsupported acceptances were 6, 1, and 2, the head-noun revision removed the modifier and possessive failures.
- **Why the model looks better.** The model's advantage is **not** evidence of better judgment. It chose `Process` for all 25 declarations, and the gate rejected 23 of those as `unstated_parent`. It avoided the `Product` trap because it never picks anything but `Process`. With 2 positives and 1 differing error, this supports no accuracy or coverage claim either way. The fair reading: the constrained model plus the gate was as precise as rules or better here, and covered no more useful claims.
- **Memory.** Prior accepted claims were retrieved 5 times through genus-head retrieval. Ontology background was retrieved 7 times, and memory changed 0 outputs. On this evidence memory neither helps nor harms.
- **Scale.** The set is small: 2 positives in 52 passages. The main coverage limit is upstream. Most definitions name a genus with no existing class (determination, estimate, movement, provision), so the precision-first system abstains on them.

## Next development work

- Make parents outside the genus table sense-checked, for example by requiring an ontology documentation or alias match, so that a false friend like `Product` cannot pass.
- Grow a reviewed genus table for common doctrine heads, with sense cues like the one for "intelligence".
- Give the model a reason to choose among parents (for example, a scoring comparison against the passage) before re-testing whether it adds anything over rules.

Keep ATP 2-01.3 reserved for a larger independent evaluation once these are in place. No training is warranted yet.
