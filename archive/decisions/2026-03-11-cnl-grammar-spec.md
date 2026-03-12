# Decision: CNL Grammar Specification

Date: 2026-03-11
Authors: Brandon Magana
Status: Accepted

---

## Decision

Adopt a six-pattern CNL grammar as the intermediate
representation for the NL→KIF pipeline.

## Core Patterns

| CNL Pattern                    | SUO-KIF Output                |
| ------------------------------ | ----------------------------- |
| Every X is a Y                 | (subclass X Y)                |
| Every X performs Y             | forall with performs relation |
| No X performs Y                | forall with negated performs  |
| Some X is a Y                  | existential instance relation |
| If S1 then S2                  | conditional, recursive        |
| Every X that performs Y is a Z | role chaining                 |

## Rationale

Six patterns cover approximately 80% of military
doctrine expression needs including:

- Subclass and instance relations
- Universal and existential quantification
- Negation
- Conditionals
- Role chaining

Keeping the grammar minimal at this stage reduces
FSM complexity and keeps the compiler correctness
proof tractable. Grammar will expand iteratively
based on benchmark document requirements.

## Class vs Instance Distinction

- Single capitalized word = SUMO class (e.g. Soldier)
- Multi-word proper noun = instance (e.g. Sergeant Smith)

Consistent with SUMO's own conventions.

## Alternatives Considered

- Larger grammar upfront — rejected, scope creep risk
- No CNL layer, direct NL→KIF — rejected, this is
  the core architectural flaw we are solving

## Open Questions

- Expressiveness ceiling — will six patterns be
  sufficient for the benchmark document?
- Deontic modality — "shall", "must", "may" in
  military doctrine not yet handled
- Potential Paper 3 contribution
