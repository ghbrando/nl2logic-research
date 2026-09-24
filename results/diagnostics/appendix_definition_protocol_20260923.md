# Head-noun gate revision and appendix protocol — 23 September 2026

## Development cycle (after the chapter 3+ run)

The [chapter 3+ run](unseen_definition_memory_20260923.md) showed six unsupported gate acceptances. Chapter 3+ has now been queried, so it counts as **development data**. The following changes were made:

- **Gate: genus head** (`genus_noun_phrase` in `src/ingest/grounding.py`). The genus noun phrase runs from the article-introduced predicate up to a boundary word (`of`, `that`, `for`, `used`, …, pronouns, verbs, and comparatives). A reviewed genus (`Process`, `IntelligenceProduct`) must be the **tail** of that phrase, so "the Army's primary process" still counts as a process. Any other parent must be the **whole** phrase, introduced by an article. A coordinated phrase matches nothing.
- **Gate: symbol sense.** The symbol may add to the defined phrase only the genus head or the parent's last word. This rule was prompted by `RiskManagementArmy` and `TroopLeadingProceduresDynamic` in the chapter 3+ outputs.
- **Proposer.** Symbols now join the defined phrase with the **genus head**. Proposals are declined for pronoun or demonstrative subjects ("This …", "There …", "The following …"), adjective predicates with no article, coordinated genera, non-kind heads such as "component", "portion", "types", and "example", heads shorter than three letters, and existing classes. Parent candidates are the fixed pool plus existing classes named by the whole genus phrase or its head.
- **Memory.** Prior accepted claims can also be retrieved when their parent names the current genus head, because literal child-word overlap never matched. They remain encoder context only.
- **Tests.** Each failure frame has a *synthetic* test sentence, not the chapter 3+ sentences themselves.

**Development results.** The saved chapter 3+ outputs were rescored under the revised gate ([scoring_head_gate](unseen_definition_cpu_20260923/scoring_head_gate/summary.json)). Every arm now has 2 label matches and **0 unsupported acceptances**; the original gate had 6, 1, and 2. The chapter 1 development results are unchanged. With the new proposer, the chapter 3+ inputs yield 22 declarations (previously 101). The rules arm accepts 4 claims, each checked by hand against its sentence: `TargetingProcess`, `RiskManagementProcess`, `KnowledgeManagementProcess`, and `TroopLeadingProceduresProcess`, each `⊂ Process`. This is development-set evidence only. The last two rules came from these passages.

## Fresh set protocol (fixed before extraction)

1. **Sources.** Run `scripts/build_unseen_definition_sources.py --appendices` on the FM 2-0 PDF. It selects every lettered appendix paragraph with a candidate head and no gate reasons. It excludes first-sentence overlaps with chapter 1, chapter 2, chapter 3+, and the seed files. Appendices have never been selected or queried. ATP 2-01.3 stays reserved and unused.
2. **Declarations.** Run `scripts/propose_declarations.py` with the revised proposer. It uses no labels.
3. **Labels.** Write provisional AI labels under the same rule as chapter 3+, validated and hash-committed before any query. The reviewer also wrote the gate, so the labels are not independent.
4. **Memory.** Prior accepted claims are the gate-accepted rules claims from chapter 1 (`partial_claim_definition_gate_cpu_20260923`) and chapter 3+ under the revised gate (`scoring_head_gate`): four claims in total. Retrieval uses child words, then the genus head.
5. **Run and report.** Run the rules, no-memory, and retrieved-memory arms in one CPU job, as before, with no training. Report the counts per arm, unsupported acceptances, abstentions, and outputs changed by memory. No rule changes are made after seeing the outputs.
