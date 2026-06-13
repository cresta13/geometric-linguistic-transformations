# Research Program: From Linguistic Transformation Vectors to Composition Diagnostics

## Core question

Can linguistic transformations be represented not only as separable classes in transformer embedding spaces, but as reusable geometric operations with meaningful composition structure?

The project currently has two paper tracks.

## Track 1: Geometric transformation vectors

Working title:

**Geometric Transformation Vectors in Transformer Embedding Spaces**

Central claim:

Transformer embedding spaces contain displacement directions that encode linguistic transformations. These directions are recoverable across multiple models and holdout regimes.

Current evidence:

- Delta vectors classify transformations well across BERT, RoBERTa, DistilRoBERTa, GPT-2, and DistilGPT-2.
- Syntax holdouts reach `1.0` accuracy across tested models, but the new representation ablation shows `y_only=1.0` as well. This is now interpreted as endpoint/surface leakage, not as deep generalization.
- Full semantic holdouts remain above chance, with accuracies around `0.82-0.89`.
- Entity and variant holdouts remain strong, especially for BERT-family models.
- Modern/larger spot-checks now support the main delta advantage:
  - BERT-large Linear SVC: `delta=0.903`, `concat=0.851`, `y_only=0.838`
  - BERT-large logistic regression: `delta=0.854`, `concat=0.760`, `y_only=0.750`
  - DeBERTa-v3-base Linear SVC: `delta=0.812`, `concat=0.776`, `y_only=0.747`
  - DeBERTa-v3-base logistic regression: `delta=0.752`, `concat=0.694`, `y_only=0.726`
- UPAT hard-holdout is a boundary result: `delta` is worse than `y_only` for BERT, RoBERTa, and GPT-2, and no UPAT delta-vs-y McNemar test is significant.
- Full-semantic pooling ablation now supports mean pooling as a defensible main choice, while showing that the decoder effect is partly pooling-dependent.

Scientific status:

This is the more mature paper. It can be written as a representation-geometry result with careful controls against template memorization.

Main risk:

Some signal may come from target-sentence artifacts rather than pure transformation geometry. The concrete high-risk case is now confirmed: `syntax=1.0` reproduces under `y_only`, so it must be interpreted as a target/surface artifact unless future controls overturn that. UPAT also shows that `delta > y_only` is not universal under small hard-holdout regimes. The paper needs strong source-only, target-only, delta-only, and paraphrase controls.

## Track 2: Signed permutation coherence for linguistic operators

Working title:

**Third-Order Signed Permutation Coherence for Linguistic Transformations in Transformer Embedding Spaces**

Central claim:

Some linguistic transformations behave like locally noncommuting operations in embedding space. Certain triples show third-order signed permutation cancellation stronger than permutation-null baselines, but the evidence supports local composition structure rather than a global Lie algebra.

Current evidence:

- Composition order matters: `AB` and `BA` often differ substantially.
- Semantic equivalence controls show lower noncommutativity for equivalent pairs than for non-equivalent pairs.
- Antisymmetry is not treated as evidence. It is a tautological implementation check because `[A,B] = delta_AB - delta_BA` implies `[B,A] = -[A,B]`.
- The non-tautological third-order signed permutation diagnostic is:

```text
S(A,B,C) = ABC + BCA + CAB - ACB - CBA - BAC
```

- The `QMT` triple shows robust cancellation across tested models:
  - BERT: ratio to permutation null `0.683`, CI `[0.677, 0.689]`
  - DistilRoBERTa: ratio `0.631`, CI `[0.627, 0.636]`
  - RoBERTa: ratio `0.623`, CI `[0.617, 0.629]`
- Decoder replication is now partially complete:
  - GPT-2 `QMT`: ratio `0.539`, CI `[0.519, 0.561]`
  - DistilGPT-2 `QMT`: ratio `0.771`, CI `[0.745, 0.797]`
  - Negation-containing triples remain mixed and model-dependent.
- Decoder pairwise composition summaries are now added, not only decoder third-order summaries.
- Multiple-testing correction over `4 triples x 5 models` supports the narrowed claim: `QMT` is the only below-null triple passing across all five tested models.

Scientific status:

This is promising but more fragile. It should be framed as a null-controlled diagnostic, not as proof of a Lie algebra or as a formal Jacobi identity.

Main risk:

Hand-written templates may induce or suppress cancellation. GPT-2/DistilGPT-2 now support `QMT` below-null cancellation, but negation behaves inconsistently across architectures. The current Lie-style templates still contain stable lexical markers, so the next experiments need grammar-driven templates, endpoint-only controls, commutator null baselines, and a focused negation analysis before expanding the operator set.

## Iterative logic

The research direction is:

1. Show transformations are encoded as recoverable displacement vectors.
2. Test whether those directions compose nontrivially.
3. Separate semantic order effects from wording artifacts.
4. Test local third-order signed permutation structure with null baselines.
5. Expand from hand-written probes to systematic grammar-generated probes.
6. Decide which claims are strong enough for publication.

## Common methodology standards

Every promoted claim should have at least one control:

- holdout split
- source-only and target-only baseline
- permutation/null baseline
- semantic equivalence control
- bootstrap confidence interval
- cross-model replication
- template de-duplication check

Negative results remain part of the research record.

## Next research plan

### Short term

1. Resolve UPAT:
   - either expand UPAT and match train sizes against the main dataset
   - or keep it explicitly as a hard negative control and narrow Track 1 claims
2. Add null baselines for exploratory UPAT alignment:
   - random-label or random-pairing Procrustes null
   - at least 1000 shuffle permutations instead of 100
3. Treat the syntax holdout as resolved for the current draft: `y_only=1.0` and layer-0 `1.0` mean the `syntax=1.0` result is a target/surface artifact unless a future redesigned split proves otherwise.
4. Convert large/modern Track 1 spot-checks into multiseed runs if Track 1 is promoted to submission.
5. Build a grammar-driven template generator for `N,Q,M,T` that produces many paraphrases without duplicate endpoints.
6. Add automated dataset validation:
   - no duplicate endpoint strings within a composition tuple
   - balanced subjects/actions
   - controlled lexical overlap
7. Add commutator null baselines for `||[A,B]||` using random or label-shuffled operations with matched norms.
8. Re-run composition and signed-permutation diagnostics on generated templates.
9. Add target-only and endpoint-only baselines for the Lie-style paper.
10. Regenerate a dated PDF packet after every major run.

### Medium term

1. Run layerwise and pooling ablations before expanding the operator set. If final-layer mean pooling is not the best representation, the expanded operator experiments should use the validated representation instead.
2. Add more linguistic operators:
   - passive voice
   - modality strength
   - evidentiality
   - aspect
   - conditionality
   - quantifier changes
3. Add paraphrase robustness with semantically equivalent endpoint variants.
4. Keep cross-model alignment as a separate future-methodology direction, not as a blocker for the current two papers. A scoped version would compare raw spaces to Procrustes-aligned spaces only after each single-model claim is stable.

### Paper milestones

1. Track 1 is arXiv-ready only when this checklist is complete:
   - syntax holdout breakdown is in the draft and interpreted as endpoint leakage
   - McNemar tests are reported in the text
   - multiseed standard deviations are reported
   - at least one prior-work baseline or comparison is written up, such as task vectors or function vectors
   - at least one model outside the original five is included as a spot-check; currently satisfied by BERT-large and DeBERTa-v3-base
   - UPAT hard-holdout is either resolved experimentally or explicitly reported as a negative boundary condition
2. Keep Track 2 as a diagnostics paper until grammar-generated templates and endpoint-only controls succeed.
3. If Track 2 survives those controls, write it as a separate paper rather than merging it into Track 1.
4. If Track 2 weakens under controls, keep it as a negative/diagnostic section in a broader research note.
