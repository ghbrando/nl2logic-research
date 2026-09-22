# Chapter 1 review packet — FM 2-0 (2023)

**100 draft passages, zero new gold labels.** The local PDF is dated
01 October 2023. The older seed benchmark describes a 2019 source, so these
records use the separate document ID `fm2-0-2023`. Do not transfer older page
references or labels to this edition without checking them.

## Files

- `review.csv`: editable annotation worksheet, with no model predictions.
- `candidates.jsonl`: immutable source records used to validate the worksheet.
- `source_pages.jsonl`: extracted text of PDF pages 17–44 for context.
- `manifest.json`: PDF hash, extraction version, page range, exclusions, and
  deterministic sampling settings.
- `triage_draft.csv` / `triage_summary.json`: a machine draft estimating which
  sentences the supported CNL fragment can express at all, with a blocker code
  per sentence. Not gold, not semantic review, and not consumable by the
  finalize script. See [TRIAGE.md](TRIAGE.md); use it as a second opinion
  while annotating, and record disagreements.

Each candidate is the first complete sentence of a numbered paragraph. `page`
is the printed page (such as `1-19`); `pdf_page` is the one-based PDF page (35).
`source_excerpt` supplies additional paragraph context. Verify extraction and
sentence boundaries against the PDF before reviewing semantics. Headers, tables,
bullet lists, and page-crossing continuations are not exhaustively represented.

Surface tags overlap: 44 relation, 14 definition, 12 modality, 5 condition,
3 negation, and 35 other. These are search heuristics, not semantic annotations.
The sample deliberately enriches these categories; it does not estimate their
prevalence in doctrine. Exact normalized matches to the three seed files and
`doctrine_real_train.jsonl` were excluded during selection.

## Review instructions

Start with [review batch 01](REVIEW_BATCH_01.md) for a focused 20-row calibration
pass. Its separate `review-batch-01.csv` keeps all annotation fields blank and
all rows as drafts; machine suggestions are in separate columns. The first two
proposals compile to five subclass edges absent from the doctrine assertions
and their transitive closure, but these examples already informed rule
development and are not a blind test. `review-batch-01-checks.json` records the
source/ontology hashes and the technical checks, not human approval.

To export explicitly reviewed rows from this batch, use the command below with
`--annotations .../review-batch-01.csv` and a new output filename. Leaving a row
as `draft` excludes it from the export. Original `review.csv` is unchanged.

For each row in `review.csv`:

1. Check the text, PDF page, paragraph, and context. If extraction needs repair,
   revise/version the candidate and worksheet together before labeling.
2. Decide whether the supported CNL fragment can express a faithful, useful
   claim. Preserve negation, direction, modality, and conditions. Do not infer
   that a source says something merely because an ontology mapping exists.
3. For a positive row, set `formalizable` to `true`, enter `pattern` and `cnl`,
   and explain partial-claim coverage or ontology assumptions in `notes`.
4. For abstention, set `formalizable` to `false`, leave CNL/pattern empty, and
   explain the reason. Keep genuinely unresolved cases as `draft`.
5. Set `status` to `reviewed` only after semantic review, fill `reviewed_by`,
   and optionally record annotation time in `review_seconds`.

Compile success alone is not semantic review. Prefer a second reviewer for
ambiguous mappings. Review without consulting model outputs; use the separate
comparison review queue later to adjudicate predictions and downstream effects.

Export completed rows from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/finalize_benchmark_review.py --candidates data/benchmarks/fm2-0/chapter1-2023-review/candidates.jsonl --annotations data/benchmarks/fm2-0/chapter1-2023-review/review.csv --output data/benchmarks/fm2-0/chapter1-2023-reviewed.jsonl
```

This validates and compiles the explicitly reviewed annotations. It never
promotes drafts, invents labels, or overwrites an existing gold file. Until rows
are reviewed, this command intentionally reports that no gold was written.

## Reproduce extraction

The preparation script requires `pdfplumber`; the checked-in packet does not.
This packet was generated with `pdfplumber==0.11.10`.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_benchmark_review.py --pdf 'src/data/ARMY FM_2-0.pdf' --doc-id fm2-0-2023 --chapter 1 --first-page 17 --last-page 44 --count 100 --exclude data/benchmarks/fm2-0/seed_positive.jsonl --exclude data/benchmarks/fm2-0/seed_abstain.jsonl --exclude data/benchmarks/fm2-0/seed_review.jsonl --exclude data/training_pairs/doctrine_real_train.jsonl --output-dir results/new-review-packet
```

Use a new output directory. The source PDF hash must match `manifest.json` to
reproduce this edition's packet. See [comparison instructions](../COMPARISON.md).
