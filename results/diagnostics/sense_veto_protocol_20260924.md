# Sense-veto training protocol — 24 September 2026

The user authorized training (lifting "do not train yet") and chose a **sense veto** over genus-contrast training. Genus-contrast training could at best tie rules. The passage-only gate never accepts more than one offered parent, and proposed class names already carry the genus, so a better parent chooser has nothing left to decide. The veto targets the one error type the gate cannot catch: a parent named by the right word but in the wrong sense. `SituationalUnderstandingProduct ⊂ Product` is an example, since SUMO `Product` is a manufactured artifact.

## Fixed before training

- **Role.** Runs only after the gate accepts a claim, and can only remove it. Input: the evidence sentence, the claim, and the parent's ontology gloss (`src/ingest/sense_veto.py`). The model scores `<claim>` against `not <claim>`, and **vetoes if log p(not claim) > log p(claim)**. There is no threshold tuning.
- **Data** (`scripts/build_sense_veto_pairs.py`). Synthetic subjects only, and every item passes the gate for its parent:
  - literal (accept) and figurative or other-sense (reject) definitions for Device, Vehicle, Weapon, Map, Key, Meeting, Report, Plan, and Organization;
  - accept-only items for Process, IntelligenceProduct, Document, Message, and Database.
  - `Sensor` is dropped because its SUMO gloss covers only software sensors.
  - **`Product` and `Cycle` never appear in training.** They form a held-out-class set that tests transfer.
- **Training.** A new LoRA adapter on `google/flan-t5-small` with `src/training/train.py` defaults (3 epochs, batch 8, learning rate 5e-4, grouped 90/10 split). It is CPU-only, runs in `save-water`'s rootless container on `spark-87dc`, and leaves the GPU to the other user's vLLM service. The generation adapter is unchanged.
- **Development check** (all queried data). Every claim the gate accepted in the chapter 1, chapter 3+ (revised gate), and appendix runs: 6 label matches and 1 unsupported claim (B-18 `Product`).
  - **Success:** the veto keeps all 6 label matches and vetoes B-18.
  - **Also reported:** synthetic held-out accuracy, and held-out-class (`Product` and `Cycle`) accuracy.

This is development evidence only. The one unqueried source is ATP 2-01.3, which stays reserved, so any claim that the model beats rules on independent data would need the user's separate approval to use it.

## v2 (fixed before v2 training)

v1 transferred poorly to the held-out classes: it vetoed only 31% of wrong-sense uses, and `Product` margins did not separate the two senses. v2 changes only the training data:
- 15 more classes with literal and figurative contrasts: Bridge, Window, Road, Ladder, Barrier, Lens, Mirror, Magnet, Wall, Pipeline, Storm, Island, Anchor, Shield, and Filter.
- **Literal** "the N of …" phrasings, so that construction alone cannot signal a veto.

`Seed` was dropped because the extractor treats "-ed" heads as verbs. The held-out `Product`/`Cycle` file is byte-identical to v1's. The training recipe, decision rule, and development claims are unchanged.

**v2 success (pre-registered):**
- **Held-out classes:** at least 60% of wrong-sense uses vetoed and at least 90% of literal uses kept, reported separately for `Product` and `Cycle`.
- **Development:** all 6 label matches kept and B-18 vetoed.

If v2 fails, the result is reported as is, with no further tuning against the held-out classes.

## v3 (fixed before v3 training)

v2 vetoed every real development claim. It had learned the synthetic surface style, not sense. v3 keeps the v2 classes, templates, recipe, and decision rule, and adds `--doctrine-style`. Each training and same-class held-out item randomly gets a doctrine-style tail (staff/echelon/operations clauses) and/or a doctrine citation such as "(JP 3-0)", **independently of its label**, so style no longer predicts accept or reject. The held-out `Product`/`Cycle` file is unchanged (same hash).

**Caveat:** this change was prompted by v2's development result, so a v3 development pass is weaker evidence than v1's pre-registered check. An independent test remains necessary.

**v3 success (pre-registered):** the same as v2. On held-out classes, at least 60% of wrong-sense uses vetoed and at least 90% of literal uses kept, for each of `Product` and `Cycle`. On development, 6/6 label matches kept and B-18 vetoed. A failure is reported as is.
