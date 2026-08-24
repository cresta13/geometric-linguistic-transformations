# Revision Notes Round 3

Date: 2026-06-13

Historical numbering note:

This file predates the 2026-08 public track renumbering. In this note, "Track 1" means the current GLT-DV delta-vector diagnostics, and "Track 2" means the current GLT-SPOT signed-permutation diagnostics. The current primary paper target is now Track 1 / GLT-STEER.

Postscript, 2026-06-23:

The later multilingual max audit revised the Track 2 narrative. The `QMT`-only claim below is historical and scoped to the original English encoder/decoder table. In the 7-language, 5-multilingual-encoder audit, all four tested triples are below signed-null in every model-language cell, with `NQM` and `QMT` strongest. Current drafts should use the multilingual framing rather than promote `QMT` as globally unique.

This note records the current response to the latest review of the research package.

## Fixed in the Current Drafts

### Antisymmetry

The Track 2 draft no longer presents antisymmetry as a discovery. It is now described as a tautological implementation check:

```text
[A,B] = delta_AB - delta_BA
[B,A] = delta_BA - delta_AB
```

Therefore `[A,B] = -[B,A]` follows for arbitrary vectors and does not provide evidence about transformer representations.

### UPAT and y_only Confounding

The Track 1 draft now explicitly reports UPAT as a hard-holdout boundary result. UPAT sometimes favors `y_only` over `delta`, and no UPAT `delta` vs `y_only` McNemar comparison is significant. The claim is narrowed:

> Delta can add useful relational information, but endpoint-only features remain a serious confounder and can dominate under smaller, harder holdouts.

### Layer-0 Syntax Result

The layer-0 syntax result is now framed as a red flag. Perfect separability at the embedding layer indicates token/form cues rather than deep semantic geometry.

### Synthetic Lie Templates

The Track 2 draft now explicitly states that the current Lie-style composition dataset is synthetic and contains stable lexical markers. The result is a controlled diagnostic, not a general natural-language operator algebra.

### Mixed Signed-Permutation Results

The Track 2 draft now explains that several triples are below null in some models, while others are above null. `QMT` is promoted only because it is the only tested triple that is below null across all five models after table-level correction.

## Still Required Before Submission

1. Add a Procrustes alignment null baseline for UPAT:
   - random-label or random-pairing alignment
   - report null distribution for F1 gain
2. Increase shuffle controls from 100 permutations to at least 1000 before reporting precise p-values.
3. Add null baselines for pairwise commutator norms `||[A,B]||`.
4. Replace hand-written Lie templates with grammar-generated paraphrase templates.
5. Run endpoint-only controls for composition endpoints.
6. Convert large/modern Track 1 spot-checks into multiseed runs if Track 1 is promoted to submission.

## Current Submission Readiness

The package is now stronger as a research record, but it should not be submitted as a main-track paper yet. The best current framing is:

- Track 1: a representation-geometry study with clear endpoint confounders and a hard-holdout boundary.
- Track 2: a diagnostic study of local signed-permutation coherence, with `QMT` as the only cross-model stable triple and negation as a composition instability.
