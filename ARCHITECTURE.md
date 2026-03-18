# Architecture

## Problem

This project is trying to solve a high-assurance autoformalization problem:

- Input: authoritative military doctrine written in natural language
- Output: a precise, prover-ready knowledge base in KIF/SUMO
- Constraint: every accepted formal statement must stay traceable to source text
- Requirement: abstention is better than hallucination

The long-term goal is not just "text to logic." The goal is a truth-grounded doctrine reasoning system that can:

1. ingest doctrine PDFs
2. formalize trustworthy facts and rules
3. build a queryable domain theory
4. support natural-language questions with proof-backed or evidence-backed answers
5. abstain when grounding or proof is insufficient

## Working Hypothesis

The primary research claim in this repo is that we can improve formalization precision by constraining generation at decode time.

Instead of letting a model freely emit logic-like text, we:

- fine-tune a small seq2seq model to produce Controlled Natural Language (CNL)
- restrict decoding with an FSM/NFA derived from the CNL grammar
- compile only valid CNL into KIF
- run a grounding gate before accepting anything into the domain

That constrained path is the main system. Everything else exists to support, test, or compare against it.

## Solution Proposal

The proposed solution is a precision-first doctrine formalization pipeline built around four layers:

1. Source preparation
   - extract sentences from doctrine PDFs
   - normalize ambiguous references when possible
   - decompose compound sentences into narrower claims
   - link surface forms to ontology-friendly terms

2. Constrained formalization
   - prompt a fine-tuned T5-family model to generate CNL
   - constrain each decode step with grammar-aware FSM transitions
   - reject unsupported continuations early instead of letting the model wander

3. Formal verification gates
   - compile CNL into KIF only when the grammar accepts it
   - check lexical and structural grounding against the source sentence
   - route each candidate to accepted, review, or reject

4. Domain construction and reasoning
   - bundle accepted statements into a prover-ready doctrine file
   - query the domain with logical retrieval and theorem proving
   - answer questions only when the system can justify the result

## Primary Pipeline

This is the architecture the repo should optimize for.

```text
Doctrine PDF
  -> sentence extraction
  -> normalization
  -> decomposition
  -> entity linking
  -> constrained NL -> CNL generation
  -> CNL compilation
  -> grounding assessment
  -> accepted / review / reject
  -> doctrine domain bundle
  -> query + proof-backed answer
```

## Main Components

### 1. Extraction

File: `src/ingest/extractor.py`

Responsibilities:

- read doctrine PDFs
- remove layout noise
- split content into traceable sentence records
- preserve page and sentence position for provenance

### 2. Preprocessing

Files:

- `src/preprocessing/normalize.py`
- `src/preprocessing/decompose.py`
- `src/preprocessing/entity_linker.py`

Responsibilities:

- rewrite or flag ambiguous references
- split compound doctrine prose into smaller claims
- map doctrine terms and acronyms into ontology-friendly surface forms

### 3. Constrained CNL Generation

Files:

- `src/fsm/cnl_fsm.py`
- `src/training/train.py`
- `src/data/generate_pairs.py`

Responsibilities:

- format model prompts
- apply grammar-derived decode constraints
- narrow generation based on sentence structure
- abstain when no credible constrained continuation exists

This is the main research contribution in the repository.

### 4. CNL Grammar and Compilation

Files:

- `src/compiler/cnl.lark`
- `src/compiler/compiler.py`

Responsibilities:

- define the allowed controlled language
- parse valid CNL deterministically
- translate parse trees into KIF

The grammar is the contract between generation and logic.

### 5. Grounding and Acceptance

File: `src/ingest/grounding.py`

Responsibilities:

- check lexical support for generated ontology terms
- apply structural checks by pattern
- block syntactically valid but semantically weak outputs
- decide whether a candidate is accepted, reviewed, or rejected

This is the current precision gate for real doctrine text.

### 6. End-to-End Ingest

File: `scripts/ingest_pipeline.py`

Responsibilities:

- run the full doctrine formalization path
- write accepted KIF, review records, rejects, and index artifacts
- provide the main operational entry point for real PDF ingestion

## Output Routing

The pipeline deliberately separates confidence levels:

- `accepted`: good enough to enter the domain automatically
- `review`: structurally useful but not trustworthy enough to auto-accept
- `reject`: unsupported, malformed, noisy, or not formalizable

This split is central to the project. High precision matters more than raw yield.

## Archived Baseline

The repo also contains an archived LLM structured-extraction baseline under:

- `archive/baselines/structured_extraction/structured_extractor.py`
- `archive/baselines/structured_extraction/test_structured_extraction.py`

That path was useful for:

- exercising downstream plumbing before the constrained model matured
- producing a baseline for comparison
- exploring silver-label generation ideas

It is not the main research architecture and should not drive current design decisions.

## Current Bottlenecks

The main open problems are:

- improving acceptance precision on real doctrine text
- learning from more realistic doctrine sentences, not just synthetic pairs
- keeping the constrained decoder expressive without reopening hallucination paths
- improving argument-level grounding so semantically weak outputs do not enter `accepted`
- building cleaner domain bundles for downstream theorem proving and QA

## Refactoring Direction

The repo should stay organized around one primary system and a small number of support layers:

- primary system: constrained T5/FSM doctrine formalization
- support layers: preprocessing, compiler, grounding, domain tooling
- archived baseline: LLM structured extraction kept only for comparison/history

When in doubt, optimize for the path that makes the constrained pipeline easier to train, evaluate, trust, and publish.
