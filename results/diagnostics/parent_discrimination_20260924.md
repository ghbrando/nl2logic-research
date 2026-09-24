# Does the adapter choose parents from the passage? — 24 September 2026

The declaration runs chose `Process` for 125 of 126 real no-memory outputs. Two CPU diagnostics, with no training and no fresh data, test whether that reflects the model or the decoding. Both use adapter `diagnostic-closed-7k-20260922-01`.

## 1. Synthetic minimal pairs

[`scripts/probe_parent_discrimination.py`](../../scripts/probe_parent_discrimination.py), run at commit `1799be1`; [summary](parent_discrimination_cpu_20260924/synthetic/summary.json).

- **Setup.** 127 defined phrases from the existing registries, each given two definitions that differ only in genus: "the process of …" versus "intelligence that is produced from …". The gate confirms which parent each variant states. The symbol is neutral (the phrase in PascalCase).
- **Scoring.** Both parents are scored by sequence log-likelihood, raw and calibrated against a genus-free sentence.

| Scoring | Item accuracy | Both members of a pair right | A constant choice would score |
| --- | --- | --- | --- |
| Raw | 77% | **54%** | 0% |
| Calibrated | 66% | 34% | 0% |

On clean contrasts, the adapter's likelihoods carry real genus signal.

## 2. Re-scoring the saved real outputs

[`scripts/rescore_saved_parents.py`](../../scripts/rescore_saved_parents.py), run at commit `c08f961`; [summary](parent_discrimination_cpu_20260924/rescore/summary.json).

- **Setup.** The exact encoder prompts of all 129 saved no-memory declaration outputs were rebuilt; each matches its saved SHA-256. Each prompt's offered parents were then scored by likelihood.
- **Beam versus likelihood.** Likelihood agreed with beam decoding on only 21 of 129. It has the opposite bias: `IntelligenceProduct` 107 times, against beam's `Process`.
- **Items with one passage-stated parent (7).** Likelihood and beam each chose the stated parent 5 times, and they missed different items:
  - Likelihood fixed 1-79 (OSINT ⊂ `IntelligenceProduct`).
  - Likelihood broke B-39 (IPOE ⊂ `Process`).
  - Neither chose the false-friend `Product` for B-18.

## Conclusion

- **The model has some genus signal**, but on real passages it is swamped by a fixed preference, whether choices are made by beam or by likelihood. The always-`Process` behavior is partly a decoding effect, but switching to likelihood selection just swaps one default for another.
- **The research question stays answered in the negative for this adapter.** It does not choose parents more accurately than the passage-stated-genus rules. And because the rules arm tries every candidate against the same gate, the model cannot cover more claims.
- **The next useful step is training**, for example on genus-contrast pairs like those in diagnostic 1. That is currently ruled out by the project instruction not to train, so it needs the user's decision.

## Addendum: can the model add coverage beyond rule candidates?

Before building the design option "let the model propose parents the rules cannot list", I checked its candidate space without the model. For the 47 source-only declarations across the three queried FM 2-0 sets, I matched each genus head against the first sentence of every SUMO class's documentation.

- **Some heads match far too many classes:** "process" matches 233 and "organization" 1,731.
- **Some match mostly the wrong senses:** "cell" includes `BloodCell`, "company" includes `AstraZeneca`, "program" includes `ComputerMenu`, and "movement" yields `AlternativeRock`, `Barrier`, … .
- **Some match nothing:** "echelon", "determination", "enabler", and "provision".

A passage-only gate cannot confirm which sense of such a class is meant. So any extra parent the model proposed from this space would either have to be abstained on or would be unsupported. The precision-safe way to add parents is a **reviewed genus-to-class table**, which is curated rules work (ideally by a subject-matter expert), not model coverage. Under the no-training constraint, this closes design option 2 as a route to beating rules.
