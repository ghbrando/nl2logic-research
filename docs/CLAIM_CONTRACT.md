# Source-claim contract for doctrine formalization

The current FM 2-0 and ATP 2-01.3 review packets score whether a **whole selected sentence** has a faithful closed CNL/KIF representation. Their labels must not be reinterpreted as labels for smaller fragments. A future packet may also evaluate **source-entailed atomic claims**, but it must identify those as a different task and use fresh source passages for evaluation.

For each proposed claim, retain:

- `doc_id`, publication edition, PDF hash, page, paragraph, and an exact supporting `source_span` within the passage;
- `coverage`: `full` when the formula preserves the whole scored sentence, or `partial` when it represents only the stated span;
- `claim_text`, `cnl`, `kif`, and the ontology terms used, with any new term *declaration* separated from the source-derived assertion;
- an AI/human reviewer attribution, a plain-language entailment rationale, and an explicit list of omitted conditions or qualifiers.

An atomic claim is acceptable only if the passage entails it without importing an unstated quantifier, existence claim, obligation, possibility, time scope, causal direction, or part-versus-kind reading. Selecting a span does not license dropping a condition that governs that span. A formula can compile and satisfy SUMO argument kinds yet still fail this semantic test.

A future packet validator must check source-span containment, closed CNL, SUMO vocabulary and arity, relation argument kinds, and exact compiled-KIF agreement. No such packet validator is implemented yet. Human or AI review must still decide entailment. AI-only annotations remain exploratory. Gold claims that merely restate a preloaded ontology axiom are excluded from positive coverage metrics; vocabulary declarations may be present as background, but source assertions used as gold may not be preloaded.

Report full-sentence and partial-claim results separately. For partial claims, also report how many source passages yielded at least one accepted claim, how many supported claims were missed, and how many accepted claims were unsupported. Abstention is correct when no claim can be represented safely. Freeze the ontology, claim schema, source selection, and labels before querying a new evaluation packet.

Example development candidate from ATP 2-01.3 paragraph 1-1: “IPB is the systematic process of analyzing...” may entail the *partial* claim “IPB is a process.” This is **not** an accepted gold label: IPB currently lacks a class declaration, the rest of the definition is omitted, and this paragraph has already been exposed in diagnostics. It cannot be reused as fresh evaluation.
