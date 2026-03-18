# FM 2-0 Benchmark Seed

Real FM 2-0 doctrine sentences annotated for NL→CNL→KIF formalization.
This is a **seed benchmark**, not a finished gold dataset.

## Files

### `seed_positive.jsonl`
Low-ambiguity examples with verified CNL and KIF annotations.

Qualification criteria:
- `formalizable: true` with non-null `pattern`, `cnl`, `kif`
- CNL compiles cleanly through the live compiler
- Ontology term choices are directly supported by `doctrine_domain.kif` or SUMO
- Sentence meaning aligns with the subclass/instance claim being made

These are the only examples that should be used as training signal or
evaluation gold until human review confirms additional examples.

### `seed_review.jsonl`
Plausibly formalizable examples that are not yet trustworthy enough for
`seed_positive`. Put examples here when:
- A CNL can be written but the ontology mapping is debatable
- The required class or relation term is missing from the closed vocabulary
- The sentence's semantic claim does not cleanly match an available pattern
- The formalization requires a doctrine extension not yet added

These have `status: "draft"`. A reviewer must resolve the noted issue
before promoting to `seed_positive`.

### `seed_abstain.jsonl`
Sentences the pipeline should correctly reject (abstain from formalizing).
These have `formalizable: false` and null `pattern`/`cnl`/`kif`.

Include:
- Unresolved pronouns with no referent
- Procedural guidance (not factual assertions)
- Deontic sentences ("should", "must") without a formalizable core claim
- Citation references

## Schema

Every record contains:

| Field | Type | Notes |
|---|---|---|
| `record_id` | string | Unique across all three files |
| `doc_id` | string | Source document identifier |
| `page` | int | Page number in source document |
| `section` | string\|null | Section identifier if known |
| `nl` | string | Original natural language sentence |
| `formalizable` | bool | Whether this sentence yields a valid formal claim |
| `pattern` | string\|null | CNL pattern: instance, subclass, relation, existential, conditional, nary |
| `cnl` | string\|null | Controlled Natural Language translation |
| `kif` | string\|null | SUO-KIF output |
| `terms` | list[string] | SUMO/doctrine ontology terms used |
| `sumo_mapping_confidence` | string\|null | high / medium / low |
| `notes` | string | Annotation rationale and known issues |
| `status` | string | draft / reviewed / accepted |
| `split` | string | seed / train / eval / abstain |

## Validation

Run the validator to check all files:

```
python scripts/validate_benchmark_annotations.py
```

To compile-check positive examples only:

```
python scripts/validate_benchmark_annotations.py --compile
```

## Source

Sentences sourced from:
`archive/baselines/structured_extraction/test_structured_extraction.py`

Original document: FM 2-0, Army Intelligence (2019).
