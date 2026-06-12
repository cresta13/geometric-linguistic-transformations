# Geometric Transformation Vectors in Transformer Embedding Spaces

## Working thesis

Transformer embedding spaces contain reusable displacement directions corresponding to linguistic transformations. These directions remain partially recoverable across prompt variants, entities, syntactic templates, and semantic holdouts.

## Evidence

- Delta-based classifiers outperform source-only controls.
- Syntax holdouts reach very high accuracy.
- Full semantic holdouts remain above chance across several models.
- Paper figures already exist in `paper/figures/`.

## Risks

- Some performance may come from target-sentence artifacts.
- Need careful separation between transformation geometry and template memorization.
- Need stronger controls against superficial lexical cues.

## Next experiments

- Expand semantic holdouts.
- Add adversarial paraphrase controls.
- Quantify how much signal is in direction versus magnitude.
