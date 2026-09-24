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
