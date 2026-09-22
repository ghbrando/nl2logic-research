# Source evidence and background axioms

Source-only extraction is the default for CNLSampler, RulesPredictor and
DoctrineGrounder, including the ingestion pipeline and comparison harness.
The existing closed vocabulary remains available, but an ontology edge cannot
supply an unstated parent to the definition router or lexical grounding gate.

The comparison CLI retains `--background-policy ontology-assisted` to reproduce
the old diagnostic baseline. The manifest records the policy. This is not an
independent extraction method: generation and validation share background edges.

This change does not prove semantic faithfulness. Aliases, lexical checks and
model weights still encode assumptions. Novel coverage currently excludes direct
subclass restatements only; it does not establish non-entailment by the ontology.

The compiler offers `require_closed=True` to reject free variables using lexical
quantifier scope. Research preflight applies this check to reviewed positive gold.
Legacy compilation still permits open formulas. Full typed claim representation,
relation argument types and acceptance-time closure enforcement remain future work.

Research preflight checks reviewed positive and abstention targets, direct
restatements and exact training overlap. A pass is only a necessary data gate.
Blind document splits, annotation quality and adequate sample sizes remain human
study-design requirements. Drafts never satisfy the reviewed-target requirement.
