# SUMO argument-kind gate — 22 September 2026

The pinned [SUMO relation-kind manifest](../../src/ontology/sumo_relation_kinds.json) records whether each relation argument expects an individual (`domain`) or a class below a specified type (`domainSubclass`). It was generated from the local SUMO checkout at `909a1ecd30949b864405206e18579ffc37318242` and is tied by SHA-256 to the existing ignored `sumo_relations.jsonl` snapshot. Strict validation fails if that snapshot changes. The manifest covers 1,504 relation records; the corresponding regeneration preserved every preexisting relation field.

`CNLCompiler.compile(..., require_argument_kinds=True)` now rejects a class token in an individual slot and an individual variable in a class slot. The comparison harness uses this gate for reviewed gold and predictions. The training-data preparation CLI enables it by default, applies it before generating paraphrases and again before selection, and omits the alphabetically adjacent subclass generator in that mode. `--allow-untyped-targets` is a legacy diagnostic escape hatch; `--allow-open-targets` also disables the kind gate. The historical training and vocabulary artifacts were not overwritten.

Applying the new gate to the prior 6,999-row closed-target diagnostic, without retraining or rewriting that artifact, gives:

| Result | Rows |
| --- | ---: |
| Retained | 2,719 |
| Rejected by argument-kind gate | 4,280 |

| Retained pattern | Rows |
| --- | ---: |
| Subclass | 1,189 |
| Binary | 148 |
| N-ary | 8 |
| Conditional | 97 |
| Existential | 1,162 |
| Negation | 115 |

Examples of rejected CNL include `agent Process AutonomousAgent`: both terms are class symbols, while SUMO `agent` expects individual process and agent arguments. A mixed relation such as `capability [ Process, ?role, ?object ]` has a class first argument and individual second and third arguments; swapping a variable into the first position or a class name into the second is rejected.

This is **argument-kind checking, not full type checking or truth checking**. It does not yet prove that a class is below the SUMO type named in a `domainSubclass` declaration, or that a bound variable belongs to the type named in a `domain` declaration. A retained synthetic target such as `partTypes Weapon MilitaryUnit` is well formed by argument kind but may be false. The corpus still contains synthetic relation assertions and ontology restatements, so the 2,719-row retained subset is **not** a new training set or a doctrine benchmark. No GPU job was started. The [source-claim contract](../../docs/CLAIM_CONTRACT.md) defines the separate full-sentence and source-entailed atomic-claim tasks needed for future annotation.
