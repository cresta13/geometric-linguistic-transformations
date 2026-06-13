# Research Program: From Linguistic Transformation Vectors to Composition Diagnostics

## Core question

Can linguistic transformations be represented not only as separable classes in transformer embedding spaces, but as reusable geometric operations with meaningful composition structure?

The project currently has two paper tracks.

## Strategic narrative

The strongest long-term story is broader than either current draft:

> Transformers may encode linguistic operators as geometric objects in a low-dimensional, partially universal transformation subspace, and this subspace may be transferable across architectures and languages.

Current results are not enough to claim this as a complete theory yet. The Procrustes transfer numbers now survive large random-label, random-pairing, random-orthogonal, and held-out-anchor alignment controls, but RISE already provides a stronger published neighboring result for spherical/geodesic semantic-syntactic transformations across languages and embedding models. The roadmap below therefore treats cross-model transfer as a stress-test and comparison track, not as an unqualified novelty claim.

Closest prior-work anchor:

- Freenor and Alvarez 2026, RISE, already demonstrates rotor-based discourse-level semantic-syntactic transformation geometry across languages and embedding models. Our work must be positioned as endpoint-controlled delta diagnostics, Procrustes/null stress testing, and ordered-composition diagnostics rather than as the first evidence for geometric linguistic transformations.

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

## Track 3: Cross-model transformation transfer

Working title:

**Stress-Testing Cross-Model Transformation Transfer in Transformer Embedding Spaces**

Central hypothesis:

Transformation geometry may be partially model-independent after low-dimensional alignment. A classifier trained on transformation deltas in one model may transfer to another model after Procrustes alignment.

Promising exploratory evidence:

- BERT to `all-mpnet-base-v2`: raw F1 around `0.15`, aligned F1 around `0.857`.
- BERT to `all-MiniLM-L6-v2`: raw F1 around `0.17`, aligned F1 around `0.754`.
- UPAT-large cross-model results show large aligned-transfer gains across several architectures.
- A scaled null audit (`N=1000` repeats per direction and null type) found that every non-identity cross-model direction stayed above random-label, random-pairing, and random-orthogonal null baselines.
- Mean observed aligned F1 was `0.684651`.
- Mean null F1 was `0.173017` for random-label, `0.148036` for random-pairing, and `0.112103` for random-orthogonal.
- No null repeat reached observed aligned F1 in any of the `30 x 3` direction/null tests, so all empirical p-values are at the `N=1000` resolution floor: `1/(1000+1)=0.000999`.
- A held-out alignment-size curve now fits Procrustes maps on `1200` auxiliary anchor texts that are disjoint from the classifier train/test texts.
- Held-out anchor mean F1 rises from `0.452046` at `25` anchors to `0.661928` at `1000` anchors, compared with raw cross-model mean F1 `0.241524` and full-anchor Procrustes mean F1 `0.684651`.
- At `1000` held-out anchors, no direction falls below its raw cross-model baseline.
- A first RISE-aware UPAT comparison now evaluates `mdv_raw`, `mdv_unit`, and a simplified spherical `rise_style` prototype baseline. Within-model target prediction cosine is high (`rise_style=0.923008`), while cross-model target prediction is harder (`rise_style` mean cosine `0.578347`, nearest-target label F1 `0.445518`) than delta-classifier transfer in aligned spaces (`0.684651` mean F1).

Current status:

This is now a serious candidate for a complementary paper, but not as a "first universal geometry" claim. RISE is stronger on cross-lingual/cross-model geometric transformation modeling. Our distinct angle is stress testing: null-controlled Procrustes transfer, held-out anchors, endpoint controls, and comparison against simpler delta/MDV/RISE-style baselines.

Required gates before promotion:

1. Add bootstrap confidence intervals and direction-family summaries for the held-out alignment and RISE-aware comparison curves.
2. Stress-test anchor-domain diversity, e.g. anchors from a different template family or natural paraphrase pool.
3. Improve the RISE-aware comparison:
   - verify the simplified `rise_style` baseline against the published RISE implementation if feasible
   - add confidence intervals
   - clarify target-prediction versus class-discrimination metrics
4. Reverse-direction transfer summary by model family, e.g. small model to large model and large model to small model.
5. Stronger architectures if feasible, such as Llama/Mistral-class embedding spaces or high-quality sentence encoders.
6. Package the result as a standalone Track 3 draft only after the RISE/MDV comparison and confidence-interval checks.

## Track 4: Transformation vectors as editors

Working title:

**From Transformation Geometry to Controllable Linguistic Editing**

Central hypothesis:

If transformation directions are real, they should not only classify transformations; they should also steer generation or hidden states toward those transformations.

Minimal experiment:

1. Use GPT-2 as the first testbed.
2. Compute a centroid delta for a transformation such as negation or question formation.
3. Inject the vector into the residual stream during generation.
4. Measure whether outputs systematically acquire the target transformation.
5. Compare against random-vector, norm-matched, and wrong-class controls.

Scientific payoff:

This would move the project from descriptive geometry to causal intervention. A positive result would connect the work to representation steering and controllable generation.

## Track 5: Cross-lingual transformation geometry

Working title:

**Language-Invariant Geometry of Linguistic Transformations**

Central hypothesis:

If transformation geometry is universal rather than English-template-specific, aligned multilingual models should show comparable transformation subspaces across languages.

Minimal experiment:

1. Use mBERT or XLM-R.
2. Build parallel transformation pairs in English and one or more non-English languages.
3. Compare delta geometry within each language.
4. Align language-specific transformation spaces with Procrustes.
5. Test whether classifiers or centroids transfer across languages.

Scientific payoff:

This is the cleanest answer to the criticism that the current effects may be English grammar or template artifacts.

## Track 6: Dimensionality of transformation manifolds

Working title:

**Effective Dimensionality of Linguistic Transformation Subspaces**

Central hypothesis:

Each transformation may occupy a low-dimensional subspace rather than a single direction or the full embedding space.

Measurements:

- PCA spectra of per-class delta matrices.
- Participation ratio of singular values.
- Classification accuracy as a function of retained dimensions.
- Sample-complexity curves for learning each transformation.

Scientific payoff:

This would quantify the complexity of each linguistic operator and explain why capacity curves keep improving with more examples.

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

1. Close methodology blockers required for any submission:
   - remove antisymmetry from the evidence narrative; already done in the draft, keep it that way
   - keep Procrustes null baselines in the evidence packet; `N=1000` random-label/random-pairing/random-orthogonal controls are now complete
   - increase shuffle/permutation controls to at least 1000, ideally 5000 for final numbers
   - add commutator norm null baselines
2. Resolve UPAT:
   - either expand UPAT and match train sizes against the main dataset
   - or keep it explicitly as a hard negative control and narrow Track 1 claims
3. Extend UPAT alignment controls beyond the completed `N=1000` null audit and held-out alignment curve:
   - bootstrap confidence intervals
   - direction-family summary
   - anchor-domain diversity check
4. Test cross-model transformation transfer as a RISE-aware stress-test paper:
   - reverse-direction transfer
   - alignment-size curve
   - random-label/null alignment controls
   - RISE/MDV-style prototype baseline
5. Treat the syntax holdout as resolved for the current draft: `y_only=1.0` and layer-0 `1.0` mean the `syntax=1.0` result is a target/surface artifact unless a future redesigned split proves otherwise.
6. Convert large/modern Track 1 spot-checks into multiseed runs if Track 1 is promoted to submission.
7. Build a grammar-driven template generator for `N,Q,M,T` that produces many paraphrases without duplicate endpoints.
8. Add automated dataset validation:
   - no duplicate endpoint strings within a composition tuple
   - balanced subjects/actions
   - controlled lexical overlap
9. Add commutator null baselines for `||[A,B]||` using random or label-shuffled operations with matched norms.
10. Re-run composition and signed-permutation diagnostics on generated templates.
11. Add target-only and endpoint-only baselines for the Lie-style paper.
12. Regenerate a dated PDF packet after every major run.

### Medium term

1. Run the GPT-2 steering-vector experiment for one transformation, starting with negation or question formation.
2. Run the cross-lingual mBERT/XLM-R transformation-transfer experiment.
3. Measure effective dimensionality of transformation subspaces:
   - participation ratio
   - PCA retention curves
   - accuracy versus retained dimensions
4. Run layerwise and pooling ablations before expanding the operator set. If final-layer mean pooling is not the best representation, the expanded operator experiments should use the validated representation instead.
5. Add more linguistic operators:
   - passive voice
   - modality strength
   - evidentiality
   - aspect
   - conditionality
   - quantifier changes
6. Add paraphrase robustness with semantically equivalent endpoint variants.

### Paper milestones

1. Track 1 is arXiv-ready only when this checklist is complete:
   - syntax holdout breakdown is in the draft and interpreted as endpoint leakage
   - McNemar tests are reported in the text
   - multiseed standard deviations are reported
   - at least one prior-work baseline or comparison is written up, such as task vectors or function vectors
   - at least one model outside the original five is included as a spot-check; currently satisfied by BERT-large and DeBERTa-v3-base
   - UPAT hard-holdout is either resolved experimentally or explicitly reported as a negative boundary condition
2. Keep Track 2 as a diagnostics paper until grammar-generated templates and endpoint-only controls succeed.
3. Promote Track 3 only if it becomes clearly complementary to RISE: null-controlled Procrustes transfer plus held-out anchors plus an explicit RISE/MDV comparison. The main narrative should be stress-testing cross-model transfer, not claiming first discovery of universal transformation geometry.
4. If steering works, write Track 4 as an intervention/controllable-generation paper.
5. If cross-lingual transfer works, it becomes the strongest version of the universality claim.
6. If Track 2 survives grammar-generated controls, write it as a separate diagnostics paper rather than merging it into Track 1.
7. If Track 2 weakens under controls, keep it as a negative/diagnostic section in a broader research note.
