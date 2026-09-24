# Passage-only definition gate and definition-child decoding — 23 September 2026

This follows the [declaration ablation](partial_claim_declaration_ablation_20260923.md). It uses the same 14 chapter 1 development paragraphs and the same provisional, label-informed registry. It is still an **upper-bound development diagnostic**, not an accuracy result. The rules below were written after seeing that ablation's six outputs.

## What changed

- **Grounding** (`assess_declared_definition` in `src/ingest/grounding.py`). A declaration contributes only its symbol, source phrase, and sense cue. The **current passage** must contain the evidence sentence, the declared phrase must be that sentence's copular subject, and the sense cue must follow it immediately. The same sentence must then state the parent as the genus. Scope and modality are checked in that sentence only. A retrieved statement is never an input, and a declaration never supplies the parent. Reasons for rejection are `undeclared_child`, `reversed_direction`, `parent_not_existing_class`, `evidence_not_in_passage`, `declaration_phrase_not_subject`, `declaration_sense_not_in_passage`, `unstated_parent`, `ambiguous_genus_sense`, and `unverified_scope`.
- **Sense of "intelligence".** In "X is intelligence derived from …", the ontology already reads this wording as a discipline (`SignalsIntelligence subclass-of IntelligenceDiscipline`). The gate therefore accepts `IntelligenceProduct` from bare "intelligence" only when the sentence itself says it "is produced". Otherwise the result is `ambiguous_genus_sense`.
- **Decoding** (`CNLSampler(definition_child=…)`). With a declaration, the only template is `<declared symbol> subclass-of <declared parent>`. Prompt n-gram classes are not offered, so an incidental word such as `Source` (from "open-source") cannot become an argument, and the direction cannot be reversed.
- **Rules-only arm.** For the same declaration, the rules arm emits the one pool parent that the gate accepts, or abstains.

## Results

Rescoring the earlier saved outputs with the new gate (no rerun): four wrong outputs were rejected, and the two biometrics outputs that matched the provisional label were accepted. The rejection reasons were `undeclared_child` (`Source subclass-of IntelligenceProduct`), `reversed_direction` (×2), and `unstated_parent`.

The [new CPU run](partial_claim_definition_gate_cpu_20260923/manifest.json) was made at commit `5092aa9`, on the same worker CPU container. Its [strict scoring](partial_claim_definition_gate_cpu_20260923/scoring/summary.json) covers 3 generating passages:

| Paragraph | Rules only | Model, no memory | Model, retrieved memory | Provisional AI label |
| --- | --- | --- | --- | --- |
| 1-79 | `OpenSourceIntelligenceProduct subclass-of IntelligenceProduct` ✓ accepted | `… subclass-of Process` ✗ `unstated_parent` | same as no memory | `… subclass-of IntelligenceProduct` |
| 1-88 | abstain | `… subclass-of Process` ✗ `unstated_parent` | same as no memory | `… subclass-of IntelligenceProduct` |
| 1-93 | `BiometricsProcess subclass-of Process` ✓ | same ✓ | same ✓ | same |

Summary of the three arms:

- **Accepted and matching the label:** rules only 2, model 1 in each arm.
- **Accepted but wrong:** 0 in every arm.
- **Memory:** changed 0 outputs.

The constrained decoder no longer produces incidental or reversed claims. The model's remaining error is choosing `Process` for both intelligence definitions, and the gate catches it. Paragraph 1-88 abstains everywhere because its wording is ambiguous in the sense above. This is a deliberate loss of coverage against the provisional label. On this tiny, label-informed set the rules arm is at least as good as the model, and nothing here suggests the model adds value yet.

## Source-only declarations

Applied to the same development candidates, the label-free proposer (`propose_declaration`) declares only `BiometricsProcess`. It declines "Open-source intelligence" and "Technical intelligence" because `OpenSourceIntelligence` and `TechnicalIntelligence` already exist as ontology classes, so a new product-sense symbol would require a sense decision the source alone does not supply. The oracle registry's other two names were therefore label-informed choices. The proposer has no way to recover them.

Unseen-passage protocol, fixed before any unseen text was extracted: see [unseen_definition_protocol_20260923.md](unseen_definition_protocol_20260923.md).
