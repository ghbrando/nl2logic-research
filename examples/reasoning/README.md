# Classification reasoning demo

This example asks whether HumanIntelligence is a subclass of Procedure.
It uses the existing reviewed FM 2-0 seed annotations, one explicit background
assumption from the local doctrine extension, and subclass transitivity.
It does not perform new model inference or independently validate the annotations.

From the repository root (use your environment's Python):

```powershell
.\.venv\Scripts\python.exe scripts/reason_ontology.py --accepted data/benchmarks/fm2-0/seed_positive.jsonl --background examples/reasoning/background.jsonl --query '(subclass HumanIntelligence Procedure)'
```

Expected result: `PROVEN`, citing:

1. `fm2-0-016`, page 18: reviewed interpretation
   `(subclass HumanIntelligence IntelligenceDiscipline)` with its source passage.
2. Background assumption `(subclass IntelligenceDiscipline Procedure)`.
3. The subclass transitivity rule.

An example of an unanswered question using known terms:

```powershell
.\.venv\Scripts\python.exe scripts/reason_ontology.py --accepted data/benchmarks/fm2-0/seed_positive.jsonl --background examples/reasoning/background.jsonl --query '(subclass HumanIntelligence WarfightingFunction)'
```

Expected result: `UNKNOWN`. Missing evidence is not an explicit negative fact.

## Prover setup

Use [Vampire's official releases](https://github.com/vprover/vampire/releases).
This integration was exercised with Vampire 5.0.1 on Windows x64. Supply
`--vampire PATH`, set `VAMPIRE_EXECUTABLE`, or install it on PATH. The CLI also
detects the project-local `vampire/5.0.1/vampire.exe` used during development.

The official Windows archive contains `vampire.exe` and `cygwin1.dll`. On the
development machine it additionally required `cygiconv-2.dll` and `cygintl-8.dll`
from Cygwin's `libiconv2` and `libintl8` packages. Keep these DLLs alongside the
executable. Release and runtime download/checksum records are in the local
installation directory. Binaries and generated run artifacts are git-ignored.
On other systems use the matching official release, not the Windows binary.

The runner uses TPTP output and `--output_axiom_names on`. A prover version that
does not return named proof inputs cannot produce a confident answer here.

## Input contract

`--accepted` reads JSONL with unique `record_id`, `kif`, and `original` or `nl`.
Existing ingest records and the reviewed seed file are supported. Explicit
draft/non-formalizable records fail validation; a status-free accepted artifact
is trusted because the caller selected it as accepted input. Page, document,
section, annotation notes, file path, and line number are retained when present.

`--background` is repeatable and actually loads its JSONL facts. Each row needs
`record_id`, `kif`, and `rationale`. These are labeled as assumptions in answers,
not presented as statements from the cited passage. No SUMO files are imported
implicitly. The demo assumption is copied from the repository's doctrine
extension and is not a new human-reviewed annotation.

## Supported semantics

Only one ground `(subclass A B)` or `(instance A B)` literal, or its explicit
`(not ...)` negation, is accepted per record/question. Names start with uppercase
ASCII letters; variables, functions, quantifiers, arbitrary predicates, and
multi-form records are rejected. All terms are preserved as quoted TPTP atoms.

The two built-in rules are subclass transitivity and instance inheritance.
There is no closed-world assumption, automatic disjointness, subclass reflexivity,
or full SUMO interpretation. In particular, `not subclass(A,B)` does not mean A
and B are disjoint. This deliberately small relational theory is not an export
of arbitrary SUO-KIF. Arbitrary natural-language questions are not supported.

## Answer contract and artifacts

The engine first checks consistency, then tries the question and its negation:

| Status | Meaning |
| --- | --- |
| `proven` | Consistent theory; a traceable proof of the question exists. |
| `disproven` | Consistent theory; a traceable proof of its explicit negation exists. |
| `inconsistent` | A traceable contradiction was found in the loaded theory; question answering stops. |
| `unknown` | Neither direction established, unsupported question, missing executable, timeout, or missing proof provenance. |

Each run writes a new directory under `results/reasoning/`: the translated
theory, evidence manifest, exact consistency/query problems, prover stdout/stderr,
and `answer.json`. The answer includes the theory hash and prover invocation
settings. The proof output contains Vampire's version. Only input axioms named
in the emitted refutation appear as supporting evidence; the full theory is
available separately. This is proof provenance extraction, not an independent
proof checker or a guarantee of minimal evidence.

Use `--json` for structured CLI output. Exit code 0 means proven/disproven,
1 means unknown/inconsistent, and 2 means invalid input or artifact setup.
`--timeout N` limits each of up to three prover calls to N CPU seconds with
an external wall-clock timeout of N+2 seconds. Existing output directories
are not overwritten.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/reasoning -q
```

Tests exercise the real prover when installed: transitive classification,
instance inheritance, explicit negative answers, open-world unknowns, direct
and derived contradictions, and the doctrine example's source/background links.
Deterministic subprocess tests cover timeouts, missing binaries, crashes, and
proofless success reports. Actual prover tests skip when no executable exists.
