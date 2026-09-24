# Unseen chapter 3+ definition run with paired memory — 23 September 2026

This run follows the [pre-registered protocol](unseen_definition_protocol_20260923.md), including its one recorded pre-query deviation (article-led declaration matching).

- **Sources.** [123 FM 2-0 chapter 3+ paragraphs](../../data/benchmarks/fm2-0/chapters3plus-definition-unseen-source/manifest.json), out of 526 scanned, selected by a label-free rule. None were used before, so they are unseen. Chapter 2 remains a previously queried set.
- **Declarations.** 101 source-only declarations were applied. 22 passages had none.
- **Labels.** [123 provisional AI labels](../../data/benchmarks/fm2-0/chapters3plus-definition-unseen-source/ai_reviewed_claims_20260923.json) (3 positive), frozen and hashed at `dad8746` before any query. The reviewer also wrote the gate, so the labels are not independent and are not expert gold.
- **Run.** One CPU job at `dad8746` in the project-owned container on worker `192.168.94.13`, with no training. The other user's GPU service was not touched. See the [predictions and manifest](unseen_definition_cpu_20260923/manifest.json) and the [scoring](unseen_definition_cpu_20260923/scoring/summary.json).

## Results

| Arm | Gate-accepted | Accepted, matches label | Accepted, **unsupported** | Label positives missed |
| --- | --- | --- | --- | --- |
| Rules only | 8 | 2 | **6** | 1 |
| Model, no memory | 3 | 2 | 1 | 1 |
| Model, retrieved memory | 4 | 2 | 2 | 1 |

- **Correct in every arm:** `TargetingProcess subclass-of Process` (3-42) and `KnowledgeManagementProcess subclass-of Process` (3-54).
- **Missed in every arm:** `PlanningHorizonPoint subclass-of TimePoint` (3-72). `TimePoint` was never offered as a parent candidate.
- **Unsupported acceptances:** `StaffKey ⊂ Key` ("a key component"), `RiskManagementArmy ⊂ Army` ("the Army's primary process"), `CorpsArmy ⊂ Army` and `BctArmy ⊂ Army` (possessive "Army's"), `StructureOfATacticalCpOrganic ⊂ Organic` ("organic to"), and `ThreeBasicFriendlyDefensiveOperationsArea ⊂ Area` ("area defense"). The rules arm accepted all six. The model accepted the `Organic` one in both arms, and the `Area` one only with memory.

## Findings

1. **The frozen gate is not precision-safe on unseen text.** When a parent has no entry in the genus table, the gate compares the parent's name with the **first word** of the predicate. Modifiers, possessives, and compound nouns therefore pass. The development set contained none of these, so the six-output development check could not reveal it. The next development cycle should require the parent to be the **head** of the genus noun phrase, and should reject possessive and adjectival first words. The proposer's head-noun symbol rule has the same flaw: it produced symbols such as `StaffKey` and `CommandersEnsureStaffSectionsProperly`. By protocol, neither rule was changed after these outputs were seen.
2. **The model does not discriminate between parents.** It chose `Process` in 100 of 101 no-memory outputs. Its better precision here comes from that default, which the gate rejects when "process" is not the genus, not from reading the passage. On this evidence the constrained model adds no demonstrated value over rules. Any gain would need better rules or a better model. Per the standing instruction, no training was started.
3. **Memory was not beneficial.** Prior accepted claims (2 development claims) were retrieved **zero** times, because no unseen candidate shared a specific word with them. Retrieved context was therefore ontology background only, supplied for 29 declarations. It changed one output (8-59), from a gate-rejected `Process` parent to the unsupported but gate-accepted `Area`, after retrieving the unrelated `(subclass HUMINTOperationsCell MilitaryUnit)`. A meaningful prior-claim comparison needs a larger store of accepted claims and a retrieval key that can match new passages, such as the genus or head noun.
4. **Abstention held for most inputs.** Of 101 generating passages, the gate rejected 98 no-memory outputs as `unstated_parent`. All 22 passages without a declaration abstained. The failure is concentrated in the head-noun path described in finding 1.

The next steps are development work only. Fix the genus-head check and the proposer head selection on development data, adding the six failure frames as synthetic tests rather than tuning against these passages. Then pre-register a fresh unseen set, such as FM 2-0 appendices or another publication. This set has now been queried.
