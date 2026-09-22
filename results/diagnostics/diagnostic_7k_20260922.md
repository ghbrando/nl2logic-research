# 7,000-pair infrastructure and data diagnostic — 22 September 2026

This is a synthetic-data diagnostic, not an independent doctrine result.
Training ran for one epoch in the rootless `save-water` research container on
`spark-87dc` (worker `192.168.94.13`). It used 6,298 training pairs and 702
grouped validation pairs from `train_balanced_7k.jsonl` (SHA-256
`77ace557ec997ad1690a0e3c5f22b03def311dfc1dd2381b6a00c338acc92e67`).
The image ID was `sha256:93c056f4bb9416eea1c678915b72958f326e1643836d4bf326ad2c2cffa55632`.
Training completed in 43.86 seconds with final teacher-forced validation loss
0.00727. The saved adapter SHA-256 is
`419ebbb61dd9b126b1545a30d88ff32f17f6524d4bb0e6537dbe43a228a2c94f`.

Full-corpus target audit: **2,372 of 7,000 targets contain unbound variables**
under `CNLCompiler.compile(..., require_closed=True)`. This includes all 1,000
`instance` targets. The exact counts and a representative failure are in
[`train_balanced_7k_audit.json`](train_balanced_7k_audit.json). No KIF mismatch
was found among the 4,628 targets that did compile as closed formulas.

On the deterministic validation split, 689/702 generated CNL strings exactly
matched their target strings. Yet only 428/702 generated strings compiled as
closed formulas. Of the 702 targets, 269 were themselves invalid, and the model
copied 261 of those invalid targets exactly. Among the 433 valid targets,
428 generated KIF formulas matched their compiled targets (98.8%). The split
shares generation templates and vocabulary with training; these numbers do not
measure doctrine generalization or semantic fidelity.

The 15-row FM 2-0 seed comparison ran rules, raw, and constrained generation
against the same checkpoint with source-only background and a 64-token budget.
After the closed-formula compiler gate was fixed, **all three accepted zero
passages**. Raw generation had ten compile errors; constrained generation also
produced outputs with unbound variables, now rejected. The seed preflight is
blocked because all reviewed positive targets restate ontology assertions.
The seed comparison is a plumbing test only.

Artifacts are retained under `/home/save-water/nl2logic-state/outputs/` on both
`spark-87dc` and the `shared-dev` controller:

- `diagnostic-7k-20260922-01/` — adapter, trainer state, validation predictions,
  `validation-v2/summary.json`, and `seed-preflight.json`.
- `seed-comparison-7k-20260922-02/` — post-fix comparison manifest, predictions,
  summary, and review queue.

The next data step is to repair or exclude open-variable synthetic targets,
then audit the regenerated file before retraining. Separately, complete the
20-row FM 2-0 calibration review and source-check/label the reserved ATP 2-01.3
packet without consulting model outputs. The ATP packet currently has zero
reviewed rows, so its preflight is blocked; see
[`atp2-01-3_draft_preflight.json`](atp2-01-3_draft_preflight.json). Only after
review, source verification, and split freezing should a formalizer comparison
be reported as an independent evaluation. Worker `spark-ce3e` remains available
for a later replication run.
