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
