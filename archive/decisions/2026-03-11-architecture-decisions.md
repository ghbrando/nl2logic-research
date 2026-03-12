# Decision: Core Architecture Choices

Date: 2026-03-11
Authors: Brandon Magana
Status: Accepted

---

## T5 over Decoder-Only Models

Chose T5 encoder-decoder over Mistral/Llama because:

- Bidirectional encoder treats task as translation
  not creative continuation
- Reduces chatty hallucinations
- More predictable for rigid grammar alignment

## LoRA over Full Fine-tuning

Chose LoRA because:

- Stable, predictable training
- Aligns probability distribution with CNL grammar
- Preserves pre-trained linguistic knowledge
- Computationally efficient on DGX Spark

## Outlines for Constrained Decoding

FSM constructed from intersection of:

- Grammar FSM — enforces CNL syntactic structure
- Vocabulary FSM — enforces SUMO term grounding

Key property: Makes ontologically ungrounded output
physically impossible regardless of model distribution.

## Lark for Compiler Implementation

- Context-free grammar definition
- Tree transformer pattern for KIF emission
- Grammar spec doubles as FSM specification for outlines
- Clean, maintainable, extensible

## Vampire for Theorem Proving

- Called as external subprocess from Python
- Non-termination treated as explicit "unknown"
  result, not failure
- Timeout bounds required — Gödel/Church-Turing ceiling

## Infrastructure Split

- Windows laptop: writing, compiler dev, Git via GitHub Desktop
- DGX Spark (100.76.51.83 via Tailscale): training,
  inference, Vampire, experiments
- Sync via GitHub — push/pull on both ends

## Build Order

1. SUMO .kif loader and term extractor
2. Compiler tests first, then compiler implementation
3. Vampire aarch64 build and subprocess wrapper
4. FSM construction and outlines integration
5. T5 + LoRA training and inference

Rationale: Layers 1-3 have no GPU dependency and
can be fully tested on Windows before touching ML.
