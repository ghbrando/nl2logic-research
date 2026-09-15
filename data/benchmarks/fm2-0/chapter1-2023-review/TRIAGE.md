# Expressibility triage — FM 2-0 (2023) chapter 1

**Machine draft, not gold.** Every verdict here was produced by reading each of
the 100 candidate sentences against the supported CNL fragment in
`src/compiler/cnl.lark`. Nothing in this file is semantic review, and
`scripts/finalize_benchmark_review.py` cannot consume it. It exists to size the
grammar's ceiling before more model training is funded, and to give the human
reviewer a starting point to confirm or overturn rather than 100 blank rows.

Regenerate with:

```powershell
.\.venv\Scripts\python.exe scripts/triage_expressibility.py --candidates data/benchmarks/fm2-0/chapter1-2023-review/candidates.jsonl --out-csv data/benchmarks/fm2-0/chapter1-2023-review/triage_draft.csv --out-json data/benchmarks/fm2-0/chapter1-2023-review/triage_summary.json
```

## Verdicts

| Verdict | Count | Meaning |
| --- | --- | --- |
| expressible | 28 | A supported pattern carries the main claim without distorting it. |
| partial | 34 | A supported pattern carries a weaker true claim; identified content is dropped. |
| blocked | 38 | No supported pattern carries the main claim without asserting something the source does not say. |

Of the 28 expressible, **8 are bibliographic** ("ATP 2-22.7 provides detailed
discussions about GEOINT"). They compile, but no user asks a knowledge base that
question. The substantive ceiling is **20 of 100**.

## The headline number

Cross-referencing the triage against the rules baseline in
`results/comparisons/2eeb7cc8e44d495b83c96c4af1ce4e5f/rules.jsonl`:

| | accepted | abstained |
| --- | --- | --- |
| expressible (28) | 3 | 25 |
| partial (34) | 0 | 34 |
| blocked (38) | 0 | 38 |

The rules baseline has **no false accepts** — it never accepted a sentence the
grammar cannot faithfully carry. Its problem is recall: it captures 3 of the 20
substantively expressible sentences, about 15% of its own ceiling.

So the grammar is not the only bottleneck. There is a roughly 6x gap between
what the fragment can already express and what the current rules produce, and
that gap is what a model arm would compete for. But the ceiling is real: even a
perfect formalizer reaches 20–28 of 100 sentences under today's grammar.

### The 17 misses

These are expressible, substantive, and abstained on:

| Frame | Count | Example |
| --- | --- | --- |
| `X capabilities consist of A, B, C` | 6 | CI, GEOINT, HUMINT, MASINT, SIGINT, intelligence PED |
| Definitional `X is a <genus> that ...` | 5 | Biometrics, forensic science, MASINT, production |
| Binary process relation | 3 | executes, is part of, drives/enables |
| Compilation / comprises list | 3 | intelligence architecture, all-source fundamentals |

One missing pattern — the consists-of frame — accounts for 6 of the 17. That is
a cheap, well-defined win, and it is the same n-ary path the arity check added
in `src/compiler/compiler.py` now validates.

## What blocks the other 72

Counts are over all 100 records, and a sentence may carry several blockers.

| Blocker | All | On blocked only | What the grammar lacks |
| --- | --- | --- | --- |
| PURPOSIVE | 13 | 3 | The claim lives in a "to ..." purpose clause |
| EVALUATIVE | 13 | 13 | crucial, difficult, significant — no formal content to capture |
| MODALITY | 13 | 11 | must / should / may / can / is responsible for |
| QUANTIFIER | 13 | 9 | most, often, primarily, comparatives, superlatives |
| META | 13 | 3 | Document self-reference and bibliography |
| TEMPORAL | 12 | 8 | continuously, throughout, in advance of, currently |
| DISJUNCTION | 7 | 4 | "or" inside the claim |
| ATTRIBUTE | 6 | 2 | Adjectival values — terms are class names or variables only |
| CONDITION | 4 | 1 | Gerund conditions, "when appropriate", "when available" |
| REFERENCE | 3 | 2 | Unresolved it / they / this |
| SCOPED_NEGATION | 3 | 2 | Negation inside a restrictive clause |
| ATTITUDE | 2 | 1 | "the Army views ... as", "JP 2-0 recognizes" |
| CARDINALITY | 1 | 1 | Numeric counts |

Two of these are worth separating from the rest.

**EVALUATIVE (13, and the single largest cause of hard blocks) is not a grammar
gap.** "The effectiveness of intelligence is measured against different
criteria" has no formal content to lose. Extending the CNL would not help; these
sentences should abstain, and a reviewer marking them `formalizable: false` is
the correct outcome, not a failure.

**MODALITY (11 hard blocks) is a grammar gap, and it is the expensive one.**
Doctrine's obligations are the part a user most wants to query — "what must the
commander establish?" — and the fragment has no deontic operator at all. Adding
one is not a grammar tweak: it needs a KIF target, an entailment story for the
prover, and training data that does not currently exist.

## The genus mismatches are a leakage symptom

Chasing the genus question below turned up the cause. `OpenSourceIntelligence
subclass-of IntelligenceDiscipline`, `TechnicalIntelligence subclass-of
IntelligenceDiscipline` and `ForensicEnabledIntelligence subclass-of
IntelligenceProduct` are asserted verbatim in `data/ontology/doctrine_domain.kif`.
The rules do not derive them from the passage; the doctrine KIF supplies the
parent to the prompt router that writes the claim *and* to the grounding gate
that licenses its direction. The sentence only has to contain "X is ...
intelligence".

That is why the genus looks wrong: nothing ever read it off the source. Six of
the eight accepted claims in the current run are such restatements, including all
three scored seed positives. The comparison harness now reports
`accepted_novel_coverage`, which excludes them — it is **0.0%** for the rules
baseline. See [COMPARISON.md](../COMPARISON.md).

Choosing a genus is therefore premature. The reviewer's labels on these
sentences, written from the passage alone, are what should settle it, and those
labels are only meaningful on records the ontology does not already answer.

## Fidelity risks found while triaging

Independent of coverage, four definitional sentences carry a genus the current
mappings appear to overwrite:

- `p35-1-79` OSINT — the source says OSINT **is intelligence** produced from
  publicly available information. The rules emit `subclass-of IntelligenceDiscipline`.
- `p36-1-88` TECHINT and `p34-1-75` MASINT — same shape.
- `p39-1-106` forensic-enabled intelligence — rules emit `subclass-of IntelligenceProduct`.

Discipline, product, and intelligence-as-content are three different things. A
reviewer should settle which genus the knowledge base commits to before these
are accepted, because downstream classification proofs inherit the choice.

Also flagged: `p38-1-98` (DOMEX) is only `partial` because the supported `not`
applies to a whole atomic assertion. Dropping "and are not publicly available"
erases exactly what distinguishes DOMEX from OSINT, so accepting the lossy form
would make the two definitions collide.

## Status: the consists-of pattern is implemented

The categorical reading was chosen: an enumerated member is a **kind of** the
named capability. `CI capabilities consist of CI teams and assigned biometric
collection equipment` now yields

```
CITeam subclass-of CICapability
BiometricCollectionEquipment subclass-of CICapability
```

Chapter-1 accepts went from **3 to 5**, with precision and false accepts on the
scored seed set unchanged (100%, zero). Two of the five enumerations resolve;
the GEOINT, MASINT, and SIGINT sentences abstain because their members are
coordinated noun phrases ("manned and unmanned platforms", "chemical,
biological, radiological, and nuclear (CBRN) detectors") that do not segment
into items. The rule refuses a partial enumeration rather than emit the members
it can resolve, since stating two of three members says less than the source
does and would disagree with a reviewer's gold label.

**Enumeration segmentation, not the genus, is now the limiting factor** for this
frame. `data/ontology/doctrine_domain.kif` deliberately omits the GEOINT/MASINT/
SIGINT member terms until that segmentation works, and asserts none of the
doctrine claims themselves — only where each term sits in SUMO — so the
extraction is not measuring its own background knowledge.

## What the consists-of pattern cost

The six "X capabilities consist of A, B, C" sentences looked like one cheap
pattern. Implementing it turned up three coupled requirements, only one of which
is a free correctness fix.

1. **Evidence direction (done).** The grounding gate licensed only
   `<child> is/are ... <parent>`. An enumeration states the same direction with
   the terms in the opposite surface order, so `CITeam subclass-of
   CounterintelligenceCapability` was routed to review with
   `unverified_subclass_direction` — and so would the identical CNL written by a
   human reviewer. `src/ingest/grounding.py` now admits the consist-of/comprise
   frame as subclass-bearing structure and verifies the child against it, with
   plural tolerance. Mereological frames stay excluded.

2. **Vocabulary (done, narrowly).** None of the enumerated terms existed in the
   29,683-term class vocabulary, so the compiler rejected the CNL outright.
   Six terms were added to `data/ontology/doctrine_domain.kif` — `CICapability`,
   `HUMINTCapability`, `CITeam`, `BiometricCollectionEquipment`,
   `HUMINTCollectionTeam`, `HUMINTOperationsCell` — each anchored to a SUMO
   parent and documented. Terms for the three unsegmentable enumerations were
   deliberately left out rather than added speculatively.

3. **Genus (decided: categorical).** "CI capabilities consist of CI teams and
   assigned biometric collection equipment" supported two readings — subclass
   (the enumeration names the members of a category) or part-of (a team is a
   constituent). Subclass was chosen: it makes "what are the HUMINT
   capabilities?" answerable and matches the project's existing doctrine-subclass
   convention. The cost is that a capability class must subsume both
   organizations and artifacts, so its SUMO parent is `Object` — a capability
   here is the set of physical things that provide it, not an abstract
   attribute. That commitment is recorded in the KIF file's header and is
   inherited by every classification proof built on it.

A reviewer who disagrees with the genus should say so before annotating, since
the rules baseline and the human gold labels must not disagree about it
systematically. It is the same kind of question as the OSINT genus above, which
is still open.

## How to use this during review

1. Work `review.csv` as the packet README describes. That file stays the gold path.
2. Use `triage_draft.csv` as a second opinion: `ceiling_verdict`, `blockers`,
   `closest_pattern`, and `loss_note` per record.
3. Disagreements are the valuable output. A `blocked` row a reviewer can in fact
   formalize means the triage is wrong; an `expressible` row a reviewer rejects
   means a pattern is unfaithful rather than merely unsupported.
4. Every `partial` row needs an explicit decision: record the loss in `notes` and
   accept the weaker claim, or abstain. Silent acceptance of a `partial` row is
   how a knowledge base ends up asserting more than its source.

See [comparison instructions](../COMPARISON.md) for the method comparison this
triage is meant to precede.
