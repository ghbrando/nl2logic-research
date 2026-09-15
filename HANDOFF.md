# Handoff — 2026-09-15

Session baseline `e8553a2` → `1a7c9a4`, four commits, 46 files, +3853/-41.
**952 passed, 1 skipped.** Branch `main`, **4 ahead of `origin/main`, not pushed.**

---

## Read this first

**The doctrine benchmark was measuring itself.** Fixing that is the whole story of
this session, and it changes what the next milestone should be.

`data/ontology/doctrine_domain.kif` is not inert background knowledge. It feeds
implied parents to *both*:

- `_infer_doctrine_subclass_plan` in `src/fsm/cnl_fsm.py`, which **writes** the
  subclass claim, and
- `_implied_doctrine_subclass_parent` in `src/ingest/grounding.py`, which
  **licenses that claim's direction** at the acceptance gate.

So a target the ontology already asserts can be produced and validated without
the source sentence carrying it. The source only has to contain "X is …
intelligence".

Measured with `scripts/audit_ontology_leakage.py`:

```
  subclass assertions      : 40
  used as training targets : 25
  used as benchmark targets:  9
  used by both             :  0
  never used               :  6

file                                            targets  restated  novel
data/training_pairs/doctrine_real_train.jsonl        27        27      0
data/benchmarks/fm2-0/seed_positive.jsonl             4         4      0
data/benchmarks/fm2-0/seed_review.jsonl               6         5      1
```

Training and benchmark targets are disjoint **from each other** — which is why
the existing exact-text leakage check passes — but both are drawn from the same
40-assertion pool that also gates acceptance. Exactly one shipped target sits
outside it: `(subclass GeneralMilitaryIntelligence IntelligenceProduct)`.

**Consequence:** the headline "100% precision / 20% coverage" describes a lookup
table being reproduced. `accepted_novel_coverage` is **0.0%**. Comparing
rules vs. raw T5 vs. constrained T5 on the current seed set would measure
nothing, for any arm. **Do not train or run the comparison as a validation of
the core research claim until there are reviewed targets outside that pool.**

---

## What changed

### 1. Semantic correctness fixes

| Area | Before | Now |
| --- | --- | --- |
| `src/preprocessing/decompose.py` | Split `or` like `and`, turning alternatives into independent assertions | Preserves disjunction, conditions, negation |
| `src/ingest/grounding.py` | Accepted reversed subclass claims | Direction must be established by the source |
| `src/ingest/grounding.py` | Rewrites could supply their own grounding evidence | Blocked |
| `src/compiler/compiler.py` | Emitted 3-arg `agent` for a binary relation | Enforces known relation arity, including nested |

### 2. Expressibility triage (`scripts/triage_expressibility.py`)

All 100 FM 2-0 chapter-1 candidates read against `src/compiler/cnl.lark`:

- **28 expressible** (20 once bibliographic sentences are set aside)
- **34 partial** — a weaker true claim is available, content is dropped
- **38 blocked**

Largest hard blockers: `EVALUATIVE` (13), `MODALITY` (11), `QUANTIFIER` (9),
`TEMPORAL` (8). **Modality is the expensive gap** — `must`/`should`/`may` have no
grammar representation at all, and that is where doctrine's obligations live.
`EVALUATIVE` is *not* a grammar gap; those sentences should abstain.

Output is `triage_draft.csv` + `TRIAGE.md` in the review packet, labeled
`claude-draft-triage` / `status: draft`. **It is a machine draft, not gold**, and
`finalize_benchmark_review.py` cannot consume it (a test pins this).

### 3. Consist-of enumeration formalization

`CI capabilities consist of CI teams and assigned biometric collection equipment`
now yields one subclass claim per member, under the **categorical reading**
(a member is a *kind of* the capability) — a decision the user approved
explicitly. Chapter-1 accepts went 3 → 5.

Three of five enumerations abstain: GEOINT, MASINT, SIGINT state members as
coordinated noun phrases (`manned and unmanned platforms`, `chemical,
biological, radiological, and nuclear (CBRN) detectors`) that do not segment.
**The rule refuses a partial enumeration rather than emit the members it can
resolve** — stating two of three says less than the source does and would
disagree with a reviewer's gold label.

`doctrine_domain.kif` gained exactly six terms for the two that work, each
anchored to a SUMO parent. It asserts **none** of the doctrine claims themselves
— a test enforces that, or extraction would be circular.

### 4. Proof-backed reasoning slice

`src/reasoning/` + `scripts/reason_ontology.py` translate a classification KIF
subset to TPTP, check consistency, and return proven / disproven / inconsistent /
unknown with source evidence and explicit background assumptions. Verified
against real Vampire. See `examples/reasoning/README.md`.

### 5. Measurement instrumentation

- `accepted_novel_coverage` in `src/eval/comparison.py` and the CLI table.
- `scripts/audit_ontology_leakage.py` — run before quoting any coverage number.

---

## Landmines

- **`restates_ontology` had a bug that would have hidden this**: it matched
  commented-out KIF assertions. Fixed in `load_asserted_subclass_pairs`.
  `src/ontology/vocab.py` guards the same way with line-anchored regexes — match
  that style if you parse KIF anywhere new.
- **Multi-claim CNL outputs** (enumerations) bypassed the subclass direction
  check entirely, because the guard used a `^…$`-anchored single-claim regex.
  Now every claim must pass; one bad claim rejects the whole output. If you add
  another multi-claim pattern, check the gate covers it.
- **Genus nouns doctrine actually uses do not exist as classes.**
  `Intelligence`, `Information`, `Analysis`, `Exploitation`, `Development`,
  `Capability`, `Mission` are all missing; `Process`, `Procedure`, `Collection`,
  `Application`, `Execution` exist. This is *why* the pipeline substitutes
  `IntelligenceDiscipline` from the lookup table instead of reading the genus off
  the sentence — the same root cause as the leakage.
- **Four definitional accepts carry a genus the source does not state.** OSINT,
  TECHINT, MASINT say "is intelligence"/"is information"; the ontology maps them
  to `IntelligenceDiscipline`. Forensic-enabled intelligence gets
  `IntelligenceProduct`. Inconsistent across identically-shaped sentences.
- **Tests write scratch dirs to the repo root**, not `tmp_path`
  (`tests/ingest/test_grounding.py:12` and five siblings). Gitignored and hidden
  from VS Code, but a fresh run recreates ~15 of them. Ten stale ones from March
  could not be removed from the sandbox (`Permission denied`); they need a normal
  terminal.

---

## Next

1. **Human review of the 100 chapter-1 rows** — `review.csv`, guided by
   `chapter1-2023-review/README.md`. This is the blocking item. It is the only
   route to targets outside the ontology pool, which is the precondition for the
   core experiment meaning anything. `triage_draft.csv` makes it confirm/correct
   rather than labeling cold.
2. Export with `finalize_benchmark_review.py`, then re-run the audit and confirm
   novel benchmark targets > 0.
3. Train on the DGX (`scripts/launch_training.sh`; no checkpoint exists —
   `models/` is empty and now gitignored), run all three arms, **compare on
   `accepted_novel_coverage`, not `accepted_coverage`**.
4. Open judgment calls for the user, not for an agent to settle alone: the
   genus for "X is intelligence …" definitions, and whether to extend the CNL
   with modality.

## Conventions

`archive/workflow.md` is the source of truth. `experiments/`, `papers/paperN/figures/`,
`archive/decisions/`, `literature/` are documented protocols, not scaffolding —
do not "clean them up." The doc also says GitHub Desktop for commits on Windows;
this session used the git CLI at the user's request.
