# Sense-veto training result — 24 September 2026

Follows the [pre-registered protocol](sense_veto_protocol_20260924.md), committed at `acbbd1e` before training.

- **Training.** A new LoRA adapter on `google/flan-t5-small` with `train.py` defaults (3 epochs, 852 steps). It trained CPU-only in `save-water`'s rootless container on `spark-87dc`; the other user's vLLM GPU service was untouched. The final validation loss was 0.036. Adapter SHA-256: `bbe08f02baed3232c512475e2bf0291a8a2fa941a8e25d265f1f552fdde9eeaf`. The adapter itself stays in `/home/save-water/nl2logic-state/outputs/partial-claim-memory-cpu-20260923-sense-veto/`.
- **Outputs.** [Evaluation outputs](sense_veto_cpu_20260924/eval/summary.json) and the [data manifest](sense_veto_cpu_20260924/data/manifest.json).
- **Decision rule.** Veto if log p(`not` claim) > log p(claim), fixed before training.

## Results

| Set | Correct claims kept | Wrong-sense claims vetoed |
| --- | --- | --- |
| **Development gate-accepted claims** (chapter 1, chapter 3+, appendix) | **6 / 6** | **1 / 1** (B-18 `Product`) |
| Synthetic, held-out subjects (training classes) | 467 / 480 (97%) | 189 / 360 (53%) |
| Synthetic, held-out classes `Product` + `Cycle` | 73 / 80 (91%) | 25 / 80 (31%) |
| — `Product` only | 36 / 40 | 9 / 40 |
| — `Cycle` only | 37 / 40 | 16 / 40 |

The pre-registered development criterion was met. Adding the veto to the rules arm removes its one unsupported acceptance on these sets and loses no correct claim.

- **Rules only:** 7 accepted, 6 correct.
- **Rules + veto:** 6 accepted, 6 correct.

## How much this shows

- **It is the first case where the model changes the result in the precision direction.** It does so as a second check on the rules and gate, not as a generator.
- **It is not reliable sense checking yet.** On synthetic `Product` definitions, which were held out of training, the median veto margin is almost the same for literal and wrong-sense uses (−2.24 against −2.05). The veto does not generally distinguish `Product` senses, so the B-18 veto (margin +1.24) may reflect something other than sense. It keeps correct claims well, but catches only about half of wrong-sense uses even for classes it was trained on.
- **The evidence is thin.** All of it is development data with one unsupported case. ATP 2-01.3, the only unqueried source, stays reserved, so there is no independent test.

## Next

- **Improve transfer on development data before any independent test:** more varied sense contrasts across many more classes, and possibly glosses that state what a class is *not*.
- **Then, with approval,** pre-register ATP 2-01.3 as a paired rules versus rules + veto comparison.

## v2 result (pre-registered criteria: failed)

v2 used the same recipe as v1 with broader synthetic contrasts ([protocol](sense_veto_protocol_20260924.md)). It trained CPU-only in 2,067 steps, with a final validation loss of 0.00012. Adapter SHA-256: `fe38543ddb9129415590f961d7a73c7a51af36763aa4b0c197d6e064301f276a`. See the [v2 evaluation](sense_veto_v2_cpu_20260924/eval/summary.json).

| Set | Correct kept | Wrong-sense vetoed |
| --- | --- | --- |
| Synthetic held-out subjects | 1080 / 1080 | 960 / 960 |
| Held-out classes `Product` / `Cycle` | 40/40 · 40/40 | 24/40 · 40/40 |
| **Development gate-accepted claims** | **0 / 6** | 1 / 1 |

- **Held-out classes: met, barely.** The median margin now separates the senses: `Product` −11.1 for literal uses against +5.5 for wrong-sense uses, and `Cycle` −13.6 against +12.6.
- **Development: failed badly.** v2 vetoes every real doctrine claim, including all 6 correct ones, at margins of +2.1 to +20.4.

The v2 veto has learned what synthetic sentences look like, not word sense. Real doctrine sentences are long, cite sources ("(JP 2-0)"), and resemble none of its training accepts, so it rejects them all. Its near-zero validation loss fits that overfitting. By the pre-registered rule, no further tuning was done against the held-out classes.

**Implications.**
- **v1's clean development result is not reliable.** It was also trained only on synthetic text, and v2 shows how far such a model can drift on real doctrine.
- **Synthetic-only training won't produce a reliable veto.** It needs accept examples that look like doctrine but do not come from the development or evaluation passages. For example, sense-verified definitions from other public doctrine, or reviewed ontology documentation sentences. That calls for a new, reviewed data source and is a decision for the user.
- **The research answer is unchanged:** there is no reliable evidence that the constrained model beats rules.

## v3 result (pre-registered criteria: failed)

v3 used the v2 data plus label-independent doctrine-style tails and citations. It trained CPU-only with a final validation loss of 0.000077. Adapter SHA-256: `d8b9f71d3b1dec13e80cdfb2965583a217a5c2f729084903f79340dfad8c1291`. See the [v3 evaluation](sense_veto_v3_cpu_20260924/eval/summary.json).

| Set | Correct kept | Wrong-sense vetoed |
| --- | --- | --- |
| Synthetic held-out subjects | 1080 / 1080 | 960 / 960 |
| Held-out classes `Product` / `Cycle` | 40/40 · 40/40 | **20/40** · 40/40 |
| **Development gate-accepted claims** | **3 / 6** | 1 / 1 |

- **Held-out classes: failed.** Only 50% of wrong-sense `Product` uses were vetoed.
- **Development: failed.** Three correct claims were vetoed: 1-93 Biometrics, 3-54 Knowledge Management, and B-44 Situation Development, all `⊂ Process`.

Randomizing style removed v2's blanket rejection of real text, but the veto still does not track sense on real doctrine. Three synthetic-data attempts have failed on real text (v1 by margins, v2 and v3 on the pre-registered criteria), so the synthetic-only route is closed. A useful veto needs real, doctrine-style, sense-reviewed training examples from outside every queried or reserved set. The DOD Dictionary was requested for this. Its site blocks scripted download, so the user needs to supply it.
