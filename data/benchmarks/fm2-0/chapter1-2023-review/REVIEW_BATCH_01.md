# Review batch 01 ? 20 source passages

**Machine-prepared worksheet, not gold.** All annotation fields remain blank and every row stays `draft`. Confirm or correct each proposal yourself; then enter the label and your reviewer name in `review-batch-01.csv`. The original 100-row worksheet is unchanged.

Purpose: calibrate annotation decisions before training. This deliberately selected development batch is not a blind test. The first two rows already informed rule development; their targets are novel relative to the doctrine assertions, but not unseen by the implementation.

Ontology novelty here checks exact asserted subclass pairs and transitive reachability in `doctrine_domain.kif` only. It does not prove novelty against all SUMO, training data, or developer knowledge. A valid compilation is not semantic approval.

Rows 1?2 propose categorical labels; 3?8 propose abstention under the current fragment; 9?20 require vocabulary, scope, or semantic decisions. Keep unresolved rows as drafts.

## 1. Paragraph 1-67 ? PDF page 33, printed page 1-17

> CI capabilities consist of CI teams and assigned biometric collection equipment.

Proposed positive under the previously approved categorical reading. Covers both members; does not assert that the list is exhaustive.

```text
CITeam subclass-of CICapability
BiometricCollectionEquipment subclass-of CICapability
```

Compiler check: passed. All 2 proposed edges are absent from both direct assertions and transitive closure of the doctrine subclass graph.

**Your decision:** confirm proposal / correct / leave unresolved.

## 2. Paragraph 1-73 ? PDF page 34, printed page 1-18

> HUMINT capabilities consist of HUMINT collection teams, HUMINT operations cells, and assigned biometric collection equipment.

Proposed positive under the previously approved categorical reading. Covers all three members; does not assert that the list is exhaustive.

```text
HUMINTCollectionTeam subclass-of HUMINTCapability
HUMINTOperationsCell subclass-of HUMINTCapability
BiometricCollectionEquipment subclass-of HUMINTCapability
```

Compiler check: passed. All 3 proposed edges are absent from both direct assertions and transitive closure of the doctrine subclass graph.

**Your decision:** confirm proposal / correct / leave unresolved.

## 3. Paragraph 1-4 ? PDF page 18, printed page 1-2

> In order to know and adapt intelligence fundamentals, intelligence professionals must read more than FM 2-0; they must be doctrinally proficient in a number of intelligence and combined arms publications, depending on their position, unit or organization, and the unit or organization’s mission or operation.

Propose abstention for the current fragment: preserve must; reading/proficiency obligations cannot become ordinary facts.

**Your decision:** confirm proposal / correct / leave unresolved.

## 4. Paragraph 1-42 ? PDF page 28, printed page 1-12

> The commander and staff must establish and support a seamless intelligence architecture, including an effective dissemination and integration plan.

Propose abstention for the current fragment: must establish is an obligation, not evidence that an architecture exists.

**Your decision:** confirm proposal / correct / leave unresolved.

## 5. Paragraph 1-7 ? PDF page 19, printed page 1-3

> Intelligence products are generally placed in one of eight production categories, based primarily on the purpose of the produced intelligence.

Propose abstention: generally, primarily, and the eight-category constraint are not represented.

**Your decision:** confirm proposal / correct / leave unresolved.

## 6. Paragraph 1-13 ? PDF page 21, printed page 1-5

> Not all echelons have the same degree of activity with other members of the intelligence enterprise, but Army intelligence professionals should understand what they receive or can access from the intelligence enterprise.

Propose abstention: not all and should cannot be replaced with universal facts.

**Your decision:** confirm proposal / correct / leave unresolved.

## 7. Paragraph 1-6 ? PDF page 19, printed page 1-3

> It is an art to describe intelligence production and dissemination and to ensure it is effectively integrated into unit planning, execution, and targeting, but it is not an exact science to execute intelligence as a function and create it as a product.

Propose abstention: metaphor and scoped negation lack a supported precise interpretation.

**Your decision:** confirm proposal / correct / leave unresolved.

## 8. Paragraph 1-8 ? PDF page 19, printed page 1-3

> The effectiveness of intelligence is measured against different criteria.

Propose abstention under the current vocabulary/fragment. Do not generalize this into a claim that evaluative statements can never be formalized.

**Your decision:** confirm proposal / correct / leave unresolved.

## 9. Paragraph 1-93 ? PDF page 37, printed page 1-21

> Biometrics is the process of recognizing an individual based on measurable anatomical, physiological, and behavioral characteristics (JP 2-0).

Keep unresolved: Biometrics is absent from the compiler vocabulary. Process is present, but a bare subclass drops the defining criterion; decide permitted partial coverage.

**Your decision:** confirm proposal / correct / leave unresolved.

## 10. Paragraph 1-104 ? PDF page 39, printed page 1-23

> Forensic science is the application of multidisciplinary scientific processes to establish facts (DODD 5205.15E).

Keep unresolved: ForensicScience is absent. The source says application of processes, not necessarily subclass Process. Validate ontology sense before choosing Application.

**Your decision:** confirm proposal / correct / leave unresolved.

## 11. Paragraph 1-38 ? PDF page 27, printed page 1-11

> Production refers to the development of intelligence through the analysis of collected information and existing intelligence.

Keep unresolved: Production and Development fail the current vocabulary check. Clarify genus and retained meaning before adding terms.

**Your decision:** confirm proposal / correct / leave unresolved.

## 12. Paragraph 1-51 ? PDF page 29, printed page 1-13

> Assess is part of the overall assessment continuing activity of the operations process.

Keep unresolved: confirm identities of Assess and assessment and the correct part relation, including class-versus-instance semantics.

**Your decision:** confirm proposal / correct / leave unresolved.

## 13. Paragraph 1-54 ? PDF page 30, printed page 1-14

> The intelligence warfighting function executes the intelligence process by employing intelligence capabilities.

Keep unresolved: executes needs a supported predicate and argument typing; do not treat class constants as process instances.

**Your decision:** confirm proposal / correct / leave unresolved.

## 14. Paragraph 1-70 ? PDF page 33, printed page 1-17

> GEOINT capabilities consist of manned and unmanned platforms and aerial and space-based collection platforms.

Keep unresolved: coordinated noun phrases need segmentation. Do not emit a partial list or add target edges to the background ontology.

**Your decision:** confirm proposal / correct / leave unresolved.

## 15. Paragraph 1-115 ? PDF page 42, printed page 1-26

> The intelligence architecture is the compilation of all relevant intelligence and communications capabilities, data repositories, organizations, supporting capabilities, and personnel necessary to ensure the successful execution of the intelligence process.

Keep unresolved: compilation of components is not automatically a categorical enumeration. Decide class membership versus part relation.

**Your decision:** confirm proposal / correct / leave unresolved.

## 16. Paragraph 1-98 ? PDF page 38, printed page 1-22

> Document and media exploitation is the processing, translation, analysis, and dissemination of collected hardcopy documents and electronic media that are under the U.S. Government’s physical control and are not publicly available.

Keep unresolved: the non-public and government-control restrictions distinguish this concept. A bare subclass would lose them.

**Your decision:** confirm proposal / correct / leave unresolved.

## 17. Paragraph 1-75 ? PDF page 34, printed page 1-18

> Measurement and signature intelligence is information produced by quantitative and qualitative analysis of physical attributes of targets and events to detect, characterize, locate, and identify targets and events; and derived from specialized, technically derived measurements and signatures of physical phenomenon intrinsic to an object or event (JP 2-0).

Keep unresolved: the source genus is information; IntelligenceDiscipline is an ontology-supplied answer, not established by this sentence.

**Your decision:** confirm proposal / correct / leave unresolved.

## 18. Paragraph 1-79 ? PDF page 35, printed page 1-19

> Open-source intelligence is intelligence that is produced from publicly available information and is collected, exploited, and disseminated in a timely manner to an appropriate audience for the purpose of addressing a specific intelligence requirement (Public Law 109-163).

Keep unresolved: the source genus is intelligence; do not silently substitute IntelligenceDiscipline.

**Your decision:** confirm proposal / correct / leave unresolved.

## 19. Paragraph 1-88 ? PDF page 36, printed page 1-20

> Technical intelligence is intelligence derived from the collection, processing, analysis, and exploitation of data and information pertaining to foreign equipment and materiel for the purposes of preventing technological surprise, assessing foreign scientific and technical capabilities, and developing countermeasures designed to neutralize an enemy’s technological advantages (JP 2-0).

Keep unresolved: the source genus is intelligence; do not silently substitute IntelligenceDiscipline.

**Your decision:** confirm proposal / correct / leave unresolved.

## 20. Paragraph 1-106 ? PDF page 39, printed page 1-23

> Forensic-enabled intelligence is the intelligence resulting from the integration of scientifically examined materials and other information to establish full characterization, attribution, and the linkage of events, locations, items, signatures, nefarious intent, and persons of interest (JP 2-0).

Keep unresolved: the source genus is intelligence; do not silently substitute IntelligenceProduct.

**Your decision:** confirm proposal / correct / leave unresolved.
