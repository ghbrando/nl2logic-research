# Closed-target 7k diagnostic — 22 September 2026

The prepared file `train_closed_7k.jsonl` contains 6,999 rows, SHA-256
`c2bd0060f32d14e87c6fc3fa86c7c369a5c1f86bfcc0f8f0b9bd414175d52461`.
Every row passes closed CNL compilation and exact stored-KIF consistency; see
[`train_closed_7k_audit.json`](train_closed_7k_audit.json). The file has no
exact overlap with the existing gold/probe check, and the reserved ATP 2-01.3
draft packet has no normalized exact-text overlap. Its [preflight](atp2-01-3_closed_7k_preflight.json)
remains blocked because zero ATP rows have human-reviewed labels.

The strict preparation path rejects open variables before synthetic paraphrases
are emitted, then checks closed CNL and KIF consistency again before balanced
sampling. Two implementations of the strict path produced byte-identical files.
The resulting six buckets contain 1,162 rows each, plus 27 pinned doctrine
subclass rows. The `instance` bucket is empty: its `?x is-a Class` targets have
an unbound variable, and replacing them with existential targets would change
their meaning. Future instance examples need named individuals or a separately
specified quantifier policy.

One epoch of LoRA training ran in the rootless `save-water` container on
`spark-87dc`, using 6,299 train and 700 grouped validation rows. Runtime was
48.06 seconds, and the final teacher-forced validation loss was 0.14096. The
adapter SHA-256 is
`65fa954c72074f675dc0cdaced4b8d839fd720dad451954fc9e958af1010f927`.
The image ID was
`sha256:93c056f4bb9416eea1c678915b72958f326e1643836d4bf326ad2c2cffa55632`.

Fresh-checkpoint generation on the deterministic validation split produced
560/700 exact CNL and compiled KIF matches (80.0%); 562/700 generated CNL
strings compiled as closed formulas (80.3%). All 700 targets were valid. Of the
138 invalid outputs, 75 failed compiler validation, 57 failed lexical parsing,
and 6 ended early. Examples show rare SUMO names being misspelled or replaced,
especially in subclass and existential targets. These synthetic templates and
vocabulary are shared with training, so the rate is not doctrine accuracy.

Rules, raw, and constrained formalizers were run on the same 15-row FM 2-0 seed
set with the new checkpoint and source-only background. All three accepted
zero passages. Its [preflight](seed_closed_7k_preflight.json) confirms that
every reviewed positive gold target restates an ontology assertion, so the
comparison remains a plumbing check rather than an independent result. The
reserved ATP packet has not been queried with the model, preserving its blind
review process.

The checkpoint and diagnostics are retained under
`/home/save-water/nl2logic-state/outputs/` on both `spark-87dc` and the
`shared-dev` controller:

- `diagnostic-closed-7k-20260922-01/` — adapter and `validation/` outputs.
- `seed-comparison-closed-7k-20260922-01/` — comparison manifest and outputs.

The next research gate is human source-and-semantic review of the FM 2-0
calibration batch and the separate ATP 2-01.3 packet. Freeze reviewed labels
and data before another model comparison. The synthetic corpus also contains
arbitrary adjacent-class subclass examples; do not interpret synthetic
validation scores as evidence of doctrinal truth.
