# Formalizer comparison protocol

## Purpose and boundaries

Compare existing rules, raw T5 generation, and constrained T5 generation before
investing in more training. This is a passage-to-one-CNL-candidate experiment,
not a PDF ingestion, preprocessing, or end-to-end QA evaluation. All arms see
identical original passages, use the same unsupported-input gate, compiler,
vocabulary, and grounding gate, and receive no gold labels in their predictor
interface. Normalization and decomposition are intentionally absent to isolate
formalization behavior.

The raw and constrained arms share the same loaded checkpoint, tokenization,
token budget, four-beam search, length penalty, and deterministic generation.
Raw generation omits the prefix constraint. Constrained failures never fall back
to raw or oracle predictions. The constrained arm also applies its existing
lexical continuation abstention rules, which are part of the method under test.

Rules use the existing prompt router only when all template slots are unique.
They share its doctrine mappings, and do not search the fallback grammar. This
is a conservative existing-rules baseline, not an optimized rule system. Seed
answers already appear in ontology assumptions and have informed development:
seed scores demonstrate plumbing, not held-out generalization.

## The seed benchmark answers itself

**The rules baseline's novel coverage on the scored seed set is 0.0%.** All three
scored accepts — `IntelligenceWarfightingFunction subclass-of WarfightingFunction`,
`HumanIntelligence subclass-of IntelligenceDiscipline`, `SignalsIntelligence
subclass-of IntelligenceDiscipline` — are claims `doctrine_domain.kif` already
asserts verbatim. Six of the eight accepts across all inputs are.

This is not a coincidence of vocabulary. The doctrine KIF supplies implied
parents to `_infer_doctrine_subclass_plan`, which generates the claim, and to
`_implied_doctrine_subclass_parent` in the grounding gate, which licenses its
direction. The source sentence only has to contain "X is ... intelligence". So
the claim is produced from the ontology and validated against the same ontology,
and the headline 100% precision / 20% coverage describes background knowledge
being handed back rather than a passage being formalized.

`accepted_novel_coverage` excludes these. Compare methods on that number. The
only two novel accepts in the current run are the chapter-1 consist-of
enumerations, and both are unscored pending review.

### The training split has the same shape

`scripts/audit_ontology_leakage.py` reports:

```
  subclass assertions            : 40
  used as training targets       : 25
  used as benchmark targets      : 9
  used by both                   : 0
  never used                     : 6

file                                            targets  restated  novel
data/training_pairs/doctrine_real_train.jsonl        27        27      0
data/benchmarks/fm2-0/seed_positive.jsonl             4         4      0
data/benchmarks/fm2-0/seed_review.jsonl               6         5      1
```

All 27 doctrine training targets and all 4 seed positives are claims
`doctrine_domain.kif` already asserts. Training and benchmark targets do not
overlap *each other* — 25 and 9, disjoint — which is why the exact-text leakage
check passes. But they partition one 40-assertion pool, and that same file
licenses acceptance at the gate.

So the doctrine split measures recall over a closed vocabulary the system
already has. A model trained on 25 of those assertions and evaluated on 9 more
is being asked to reproduce a lookup table, not to formalize a passage. Exactly
one shipped target is outside the pool:
`(subclass GeneralMilitaryIntelligence IntelligenceProduct)`.

Run the audit before quoting any coverage number:

```powershell
.\.venv\Scripts\python.exe scripts/audit_ontology_leakage.py --training data/training_pairs/doctrine_real_train.jsonl --benchmark data/benchmarks/fm2-0/seed_positive.jsonl --benchmark data/benchmarks/fm2-0/seed_review.jsonl
```

Fixing this properly means scoring against passages whose answers the ontology
does not already contain — which is what the chapter-1 review packet is for.

## Known ceiling on chapter 1

A draft triage of the 100 chapter-1 candidates against `src/compiler/cnl.lark`
puts 28 as expressible (20 once bibliographic sentences are set aside), 34 as
partial, and 38 as blocked. The rules baseline accepts 5 of those 28 with no
false accepts, so the gap a model arm competes for is recall, not precision, and
the best achievable coverage under today's grammar is roughly 20-28 of 100.
Read [chapter1-2023-review/TRIAGE.md](chapter1-2023-review/TRIAGE.md) before
interpreting any coverage number from this harness.

## Run now without a checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/compare_formalizers.py --methods rules --benchmark data/benchmarks/fm2-0/seed_positive.jsonl --benchmark data/benchmarks/fm2-0/seed_abstain.jsonl --benchmark data/benchmarks/fm2-0/chapter1-2023-review/candidates.jsonl --training-data data/training_pairs/doctrine_real_train.jsonl --training-data data/training_pairs/train_balanced_50k.jsonl
```

The original 15 reviewed seed records are scoreable; the 100 new drafts are
reported only as unscored routing diagnostics. Review them using the
[annotation worksheet](chapter1-2023-review/README.md).

## Run all three methods

```powershell
.\.venv\Scripts\python.exe scripts/compare_formalizers.py --model-path models/flan-t5-small-cnl --benchmark data/benchmarks/fm2-0/chapter1-2023-reviewed.jsonl --training-data data/training_pairs/train_balanced_50k.jsonl --training-data data/training_pairs/doctrine_real_train.jsonl
```

Supply the checkpoint's actual training files, not merely the example paths.
Inference needs the project's torch/transformers/peft dependencies. The default
methods are all three. Without a checkpoint, the model arms are `unavailable`,
their metrics are null, and the command exits 1 after preserving available
results. Use `--methods rules` for an explicitly rules-only run. Existing output
directories are never overwritten. Invalid benchmark data exits 2.

Replace the draft input with the reviewed export; do not pass both files with
the same record IDs. Training-split records and normalized exact NL overlaps
with supplied training files are excluded from scored denominators. This does
not detect paraphrases, ontology leakage, or prior manual tuning. Without
training files the run records that leakage was not checked. Reserve a separate,
untouched chapter/document for a future blind test.

## Reports

Each new directory under `results/comparisons/` contains:

- `manifest.json`: hashes of inputs, checkpoint files (when available), relevant
  code and knowledge files; generation budget; dependency versions; protocol;
  training overlap details.
- `benchmark_snapshot.jsonl`: exact input records.
- `rules.jsonl`, `raw.jsonl`, `constrained.jsonl`: per-record predictions, routing,
  errors, latency, and score eligibility for available arms.
- `summary.json`: aggregate, per-document, and per-surface-phenomenon metrics.
- `review_queue.jsonl`: gated review cases, accepted mismatches, and accepted
  unscored candidates, with fields for adjudication, time, and affected queries.

### Metric definitions

| Metric | Definition |
| --- | --- |
| Accepted exact precision | Accepted reviewed-positive KIF matches / all accepted scored predictions. Null if nothing accepted. |
| Accepted coverage | Accepted scored predictions / all scored records. Includes ontology restatements; not the figure to compare methods on. |
| Accepted restating ontology | Accepted predictions whose every subclass claim is already asserted in `data/ontology/doctrine_domain.kif`. |
| Accepted novel coverage | Accepted scored predictions that are NOT ontology restatements / all scored records. **This is the comparison figure.** |
| Correct accepted positive coverage | Accepted exact matches / reviewed formalizable positives. |
| Abstention accuracy | True abstentions / reviewed abstention examples. Runtime and compile errors are not abstentions. |
| False accepts on abstention gold | Accepted predictions for examples reviewed as requiring abstention. |
| Review queue size | Number routed to review by the grounding gate; the adjudication file additionally includes mismatches and unscored accepts. |

Exact KIF agreement is only a reproducible proxy for correctness. Equivalent or
valid partial translations can differ from the single reference; adjudicate
mismatches before calling them semantic errors. No confidence claim should be
based on a handful of accepted examples. Runtime includes shared gates and
generation, excludes model loading, and has no hardware/warmup normalization.

Human review effort and downstream-answer impact are not fabricated from these
proxies. `human_review_seconds` stays null until measured. Use the adjudication
queue to record `review_seconds`, `changes_downstream_answer`, and `affected_query`.
The harness does not yet run theorem-prover QA comparisons automatically.

## Initial diagnostic finding

On the existing seed, the rules baseline accepted 3 of 4 positives with exact
KIF agreement, sent the remaining positive to review, and abstained on all 11
abstention examples. The review case contains “on or about,” triggering the
current conservative scope gate. This shows a coverage cost worth investigating
on independent data. It does not establish that rules outperform either model.

On the 100 new unreviewed passages, rules produced 3 accepted candidates and 97
abstentions. Their correctness is unscored. No local checkpoint was available,
so there are no raw/constrained model accuracy results from this implementation.
