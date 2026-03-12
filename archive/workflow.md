# NL2Logic Research Workflow

Authors: Brandon Magana, Eli Manning
Created: 2026-03-11

---

## Daily Loop (30 min minimum)

1. Review open questions from yesterday
2. Complete one concrete task — code, writing, or reading
3. Log what was done and what is blocked

## Weekly Loop

1. Review paper pipeline tracker in README
2. Draft or revise one paper section
3. Read and log one piece of literature in literature/reading-log.md
4. Personal architecture review — has anything challenged current assumptions?

## Monthly Loop

1. Full architecture review
2. Update archive/decisions/ with new choices made
3. Assess venue deadline alignment for active papers

## Writing Workflow

- Week N:   Build or experiment
- Week N+1: Write what you built while fresh
- Week N+2: Revise
- Week N+3: Back to building

Rule: Never let more than two weeks pass between
doing something and writing about it.

## Conversation Archive Protocol

1. Export every significant AI research conversation
2. Save to archive/conversations/YYYY-MM-DD-topic.md
3. Write one paragraph summary in your own words
   without looking at the conversation
4. If you cannot write the paragraph, you do not
   understand it yet — re-read before archiving

## Experiment Protocol

Every experiment gets a dated folder under experiments/:

- config.json   — exact parameters used
- results.json  — raw results
- notes.md      — interpretation and next steps

All experiments must be reproducible from config alone.

## Version Control Rules

- Every architecture decision gets a dated markdown
  file in archive/decisions/
- Commit messages must be descriptive — not "update" or "fix"
- Eli and Brandon both review PRs for paper sections
- Never commit directly to main for paper drafts — use branches
- GitHub Desktop for all commits on Windows
- Git CLI on DGX Spark for experiment pushes

## LaTeX Rules

- All papers written in LaTeX from day one
- Use ACM or IEEE template per target venue
- Figures go in papers/paperN/figures/
- One sentence per line in .tex files for clean diffs
