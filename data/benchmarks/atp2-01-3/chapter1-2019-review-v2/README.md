# Reserved review packet: ATP 2-01.3 (2019), chapter 1

This packet contains **40 draft passages and zero reviewed labels** from PDF
pages 15–30. It is reserved for a future independent document evaluation; do
not train on it, tune rules against it, or inspect model predictions while
annotating. The source is the U.S. Army's public [ATP 2-01.3 PDF](https://home.army.mil/wood/8915/5751/8365/ATP_2-01.3_Intelligence_Preparation_of_the_Battlefield.pdf).
The downloaded PDF is kept outside Git at `tmp_heldout/ATP_2-01.3_2019.pdf`.
Its SHA-256 is `10a08e6ff0779c7d6c37b4d05f3ab813fcbd86f78d3eefc5348d19e42511827d`;
verify this hash before using a replacement copy.

`candidates.jsonl` freezes record IDs, passage text, context, and draft status.
`review.csv` is the blank human annotation worksheet. `source_pages.jsonl`
contains extracted page text for tracing, and `manifest.json` records the
selection settings and exclusion hashes. The 40 rows are a seeded,
phenomenon-enriched selection from 59 numbered-paragraph candidates, not a
representative random sample. Exact normalized text matches to the listed
training and FM 2-0 benchmark files were excluded; paraphrase and conceptual
overlap have not been checked.

Before labeling a row, verify its sentence, paragraph, and printed page against
the PDF. The source PDF's text layer contains some replacement characters (`�`)
and the extractor can join text across paragraph boundaries. Repair and version
the candidate and worksheet together if needed. Then judge whether the current
CNL grammar expresses a faithful claim, including direction, quantification,
conditions, negation, and modality. For a positive row, fill `formalizable`,
`pattern`, `cnl`, and explanatory `notes`; for abstention, set `formalizable` to
`false` and explain why. Change `status` to `reviewed` and fill `reviewed_by`
only after checking the source and semantics. Keep unresolved rows as `draft`.

The existing 20-row [FM 2-0 calibration batch](../../fm2-0/chapter1-2023-review/REVIEW_BATCH_01.md)
is for developing annotation practice. It already influenced rules and must
remain separate from this reserved document. Freeze this packet's annotations
and training inputs before running the research preflight or formalizer
comparison on reviewed exports.
