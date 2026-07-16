# Research Program: From Linguistic Transformation Vectors to Composition Diagnostics

## Core question

Can linguistic transformations be represented not only as separable classes in transformer embedding spaces, but as reusable geometric operations with meaningful composition structure?

This repository develops **GLT** (**Geometric Linguistic Transformations**), a research program for probing whether linguistic transformations appear as reusable geometric objects in transformer embedding spaces.

Current tracks:

- **GLT-DV**: delta-vector diagnostics with endpoint controls.
- **GLT-SPOT**: signed-permutation operator tests for ordered composition.
- **GLT-XFER**: cross-model transformation-transfer stress tests.
- **GLT-STEER**: activation-steering interventions that inject transformation vectors back into generative models.
- **GLT-MOLT**: planned matrix/operator extension motivated by Linear Relational Decoding.
- **GLT-AFFECT**: planned graded affective-transformation geometry, starting with text-only emotional polarity scales and reserving sensory grounding claims for independent perceptual data.

## Strategic narrative

The strongest long-term story is broader than either current draft:

> Transformers may encode linguistic operators as geometric objects in a low-dimensional, partially universal transformation subspace, and this subspace may be transferable across architectures and languages.

Current results are not enough to claim this as a complete theory yet. The Procrustes transfer numbers now survive large random-label, random-pairing, random-orthogonal, and held-out-anchor alignment controls, but RISE already provides a stronger published neighboring result for spherical/geodesic semantic-syntactic transformations across languages and embedding models. The roadmap below therefore treats cross-model transfer as a stress-test and comparison track, not as an unqualified novelty claim.

Closest prior-work anchor:

- Freenor and Alvarez 2026, RISE, already demonstrates rotor-based discourse-level semantic-syntactic transformation geometry across languages and embedding models. Our work must be positioned as endpoint-controlled delta diagnostics, Procrustes/null stress testing, and ordered-composition diagnostics rather than as the first evidence for geometric linguistic transformations.
- Xia and Kalita 2025, Linear Relational Decoding of Morphology in Language Models, shows that some linguistic relations can be decoded by relation-specific Jacobian-derived matrix operators, with strong morphology results across GPT-J, Llama-7b, and multilingual morphology. This is a key motivation for an operator-valued version of Track 2: delta vectors are not the only plausible representation of linguistic transformations.

## Track 1: Geometric transformation vectors

Working title:

**Geometric Transformation Vectors in Transformer Embedding Spaces**

Central claim:

Transformer embedding spaces contain displacement directions that encode linguistic transformations. These directions are recoverable across multiple models and holdout regimes.

Current evidence:

- Delta vectors classify transformations well across BERT, RoBERTa, DistilRoBERTa, GPT-2, and DistilGPT-2.
- The cleanest multiseed claim is probe-dependent: Linear SVC shows reproducible `delta > y_only` across all original models, with positive 95% seed-level intervals; logistic regression is mixed for GPT-2 and RoBERTa.
- Syntax holdouts reach `1.0` accuracy across tested models, but the new representation ablation shows `y_only=1.0` as well. This is now interpreted as endpoint/surface leakage, not as deep generalization.
- Full semantic holdouts remain above chance, with accuracies around `0.82-0.89`.
- Entity and variant holdouts remain strong, especially for BERT-family models.
- Modern/larger spot-checks now support the main delta advantage:
  - BERT-large Linear SVC: `delta=0.903`, `concat=0.851`, `y_only=0.838`
  - BERT-large logistic regression: `delta=0.854`, `concat=0.760`, `y_only=0.750`
  - DeBERTa-v3-base Linear SVC: `delta=0.812`, `concat=0.776`, `y_only=0.747`
  - DeBERTa-v3-base logistic regression: `delta=0.752`, `concat=0.694`, `y_only=0.726`
- UPAT hard-holdout is a bounded-claim result: `delta` is worse than `y_only` for BERT, RoBERTa, and GPT-2, and no UPAT delta-vs-y McNemar test is significant. This suggests that the main delta advantage captures transformation type better than absolute transformation identity under hard sentence-pair splits.
- Full-semantic pooling ablation now supports mean pooling as a defensible main choice, while showing that the decoder effect is partly pooling-dependent.

Scientific status:

This is the more mature paper. It can be written as a representation-geometry result with careful controls against template memorization.

Main risk:

Some signal may come from target-sentence artifacts rather than pure transformation geometry. The concrete high-risk case is now confirmed: `syntax=1.0` reproduces under `y_only`, so it must be interpreted as a target/surface artifact unless future controls overturn that. UPAT bounds the positive claim rather than merely weakening it: `delta > y_only` is not universal under small hard-holdout regimes, which suggests that delta vectors are more reliable for transformation-type geometry than for absolute sentence-pair identity. The logistic-regression rows show the same caution from a different angle: the Track 1 result is currently strongest as a Linear SVC/margin-based probe result, not as a classifier-invariant law. The paper needs strong source-only, target-only, delta-only, and paraphrase controls.

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
- In the original English encoder/decoder table, multiple-testing correction over `4 triples x 5 models` supports the narrowed claim that `QMT` is the only below-null triple passing across all five tested models.
- A grammar-generated pairwise composition control now tests `720` generated `N,Q,M,T` rows across BERT, DistilRoBERTa, and RoBERTa.
- In that grammar-control probe, observed relative commutator norms remain below all `N=1000` same-pair, any-pair, and norm-matched null means at the current `p=0.000999` resolution floor.
- But endpoint-only and delta-only pair-label controls are essentially perfect (`macro F1 ~= 1.0`), so this is not endpoint-independent algebraic evidence yet.
- A 2026-06-23 multilingual max audit now extends the signed-permutation probe to 7 languages and 5 multilingual encoders:
  - languages: English, Spanish, French, German, Russian, Chinese, Arabic
  - models: paraphrase-multilingual-mpnet-base-v2, LaBSE, multilingual-e5-large, BGE-M3, and mBERT
  - all four tested triples are below signed-null in all `35/35` model-language cells
  - global mean ratios to signed-null: `NQM=0.580`, `QMT=0.620`, `NQT=0.701`, `NMT=0.772`
  - source-only held-out-language controls are chance-like (`macro F1=0.0476`), but endpoint/delta/commutator controls remain high, so endpoint artifacts are not solved
  - cross-language centroid consistency is moderate and high-variance (`mean cosine ~= 0.32`)
- A 2026-06-27 extended multilingual GLT-SPOT audit scales the same diagnostic to 7 models, 7 languages, 96 templates per language, PCA dimension 128, and 5000 signed-null repeats:
  - added XLM-RoBERTa and DistilBERT multilingual to the previous 5-model audit
  - all four tested triples remain below signed-null in all `49/49` model-language cells
  - global mean ratios to signed-null: `NQM=0.571`, `QMT=0.620`, `NQT=0.700`, `NMT=0.777`
  - cross-language centroid consistency remains moderate and high-variance (`triple_signed mean cosine ~= 0.255`)
  - source-only held-out-language controls remain chance-like (`macro F1=0.0476`), while endpoint/delta/commutator controls remain high
  - this strengthens the controlled signed-permutation coherence claim, but does not solve endpoint-derived leakage
- A 2026-06-27 third-order endpoint-control audit tests whether triple identity is recoverable from third-order endpoints under held-out-language transfer:
  - models/languages/templates match the extended multilingual audit (`7 x 7 x 96`)
  - source-only remains weak (`macro F1=0.100`, chance for four triples is `0.250` accuracy but macro F1 is low due to degenerate predictions)
  - signed endpoint/delta sums are highly predictive (`macro F1=0.708`)
  - endpoint concatenation is stronger (`macro F1=0.835`)
  - delta concatenation is strongest (`macro F1=0.876`)
  - this confirms that triple identity is strongly encoded in endpoint/delta geometry, so the current GLT-SPOT evidence is not endpoint-independent
- A 2026-06-28 new-model GLT-SPOT check tests newer multilingual embedding models at 48 templates per language and 2000 signed-null repeats:
  - `intfloat/multilingual-e5-large-instruct`: all four triples below null in all `7/7` language cells (`NQM=0.615`, `QMT=0.649`, `NQT=0.739`, `NMT=0.780`)
  - `Qwen/Qwen3-Embedding-0.6B`: all four triples below null in all `7/7` language cells (`NQM=0.539`, `QMT=0.639`, `NQT=0.692`, `NMT=0.727`)
  - Qwen required casting pooled embeddings to `float32` before NumPy conversion because the model returns `bfloat16` tensors on CPU
  - `Alibaba-NLP/gte-multilingual-base` and `jinaai/jina-embeddings-v3` are not reported here; the exploratory combined run terminated during `gte-multilingual-base` before producing a completed checkpoint
- A 2026-06-28 new-model endpoint-subspace residualization audit then removes endpoint-derived linear rowspaces from the same newer model family:
  - models: `intfloat/multilingual-e5-large-instruct`, `Qwen/Qwen3-Embedding-0.6B`
  - templates per language: `96`
  - probe evidence confirms endpoint-derived leakage is present (`triple_label_from_single_endpoint_delta macro F1=0.781`)
  - after removing triple-label, endpoint-position, and cyclic-sign rowspaces together, all triples remain below exact sign-null in all `14/14` model-language cells
  - global raw versus remove-all ratios:
    - `NQM`: `0.556 -> 0.559`
    - `QMT`: `0.621 -> 0.629`
    - `NQT`: `0.699 -> 0.704`
    - `NMT`: `0.739 -> 0.738`
  - this is the strongest current evidence that the GLT-SPOT below-null signal is not only a linear endpoint-label artifact, while still falling short of a formal Lie-algebra proof
- A 2026-06-29 9-model endpoint-subspace residualization audit extends this linear-control layer to older and newer multilingual embedding models:
  - models: paraphrase-multilingual-mpnet-base-v2, LaBSE, multilingual-e5-large, BGE-M3, mBERT, XLM-RoBERTa, DistilBERT multilingual, multilingual-e5-large-instruct, and Qwen3-Embedding-0.6B
  - languages: English, Spanish, French, German, Russian, Chinese, Arabic
  - templates per language: `96`; PCA dimension: `128`
  - after removing triple-label, endpoint-position, and cyclic-sign rowspaces together, all four triples remain below exact signed-null in all `63/63` model-language cells
  - global raw versus remove-all ratios:
    - `NQM`: `0.549 -> 0.547`
    - `QMT`: `0.602 -> 0.615`
    - `NQT`: `0.684 -> 0.685`
    - `NMT`: `0.756 -> 0.755`
  - endpoint-derived information remains measurable (`triple_label_from_single_endpoint_delta macro F1=0.765`), so the result should be read as robustness to these linear subspace removals rather than endpoint independence
- A 2026-06-29/30 9-model GLT-MOLT affine/operator audit then tests whether learned operator maps give a more Lie-adjacent signal:
  - models and languages match the 9-model multilingual stress-test family
  - templates per language: `160`; PCA dimension: `128`; random-subspace nulls: `1000` in the confirmation run
  - simple additive displacement vectors outperform learned linear/affine operators at target prediction (mean target cosine `0.903` additive versus `0.738` linear and `0.695` affine), suggesting that endpoint differences capture target movement more naturally than parametric operator regression in the current setting
  - learned matrix commutators are not closed exactly, but closure residuals are systematically below random-subspace nulls:
    - linear pair residual range: `0.964-0.985` versus random null `~0.9999`
    - affine pair residual range: `0.958-0.987` versus random null `~0.9999`
  - relative Jacobi-like operator norms are low across triples (`linear mean 0.067`; `affine mean 0.064`)
  - interpretation: this is the strongest GLT-MOLT/operator-valued diagnostic so far, but it is still evidence for weak closure-like compression in learned PCA-space maps, not a formal Lie algebra
- A 2026-07-01 GLT-MOLT ridge sweep tests whether the operator-valued signal is stable across regularization:
  - ridge alphas: `0.1`, `1.0`, `10.0`, `100.0`; models/languages/templates match the 9-model MOLT audit
  - additive target prediction is unchanged (`mean target cosine 0.903`)
  - linear/affine target prediction is best around `alpha=10` (`linear 0.738`, `affine 0.695`) and worsens at `alpha=100`
  - closure residuals improve monotonically with stronger ridge smoothing:
    - linear: `0.995 -> 0.991 -> 0.975 -> 0.882`
    - affine: `0.994 -> 0.991 -> 0.970 -> 0.855`
  - Jacobi-like norms also generally improve, especially affine at `alpha=100` (`0.042`)
  - interpretation: the closure-like signal is robust across alphas, but algebraic cleanliness is partly regularization-sensitive; future MOLT nulls must be norm-matched or shrinkage-matched before making stronger Lie-style claims
- A 2026-07-02 GLT-MOLT matched-null audit directly tests that concern for `alpha=10` and `alpha=100`:
  - null controls: random-subspace, Gaussian norm-matched operator maps, and signed-permutation matched operator maps
  - observed closure remains below all three null families
  - `alpha=10`: affine closure `0.970`, linear closure `0.975`, matched null means `~0.9998`
  - `alpha=100`: affine closure `0.855`, linear closure `0.882`, all mean empirical p-values hit the `N=1000` resolution floor `0.000999`
  - target prediction remains worse at `alpha=100` than at `alpha=10`, so the result separates algebraic compression from endpoint reconstruction
  - interpretation: this strengthens GLT-MOLT as an operator-closure diagnostic, but it still supports only weak closure-like compression in ridge-regularized PCA-space maps, not a formal Lie algebra
- A 2026-07-03 GLT-MOLT spectral-null audit then tests the strongest current shrinkage/spectrum concern at `alpha=100`:
  - null control: singular-spectrum matched commutator matrices generated by random Givens row/column rotations
  - observed closure remains below the spectral null:
    - affine closure `0.855` versus spectral null `0.99985`, mean empirical p `0.00333`
    - linear closure `0.882` versus spectral null `0.99985`, mean empirical p `0.00347`
  - interpretation: the closure-like compression is not explained by the commutator singular-value spectrum alone, though model-level magnitudes remain uneven and the result still supports only a controlled operator-closure diagnostic
- A 2026-07-08 compact PCA sensitivity check tests whether the spectral-null signal survives alternate PCA dimensions on five stable multilingual encoders:
  - PCA-64: affine closure `0.8285` versus spectral null `0.9996`, mean empirical p `0.00332`; linear closure `0.8701` versus spectral null `0.9995`, mean empirical p `0.00335`
  - PCA-128: affine closure `0.8153` versus spectral null `0.9999`, mean empirical p `0.00332`; linear closure `0.8581` versus spectral null `0.9999`, mean empirical p `0.00332`
  - interpretation: this supports PCA-dimensional robustness at `64` and `128`, but it is still a compact stable-model sensitivity check rather than a complete PCA sweep; PCA-256 should be run only as a separate smaller job if needed
- A 2026-06-24 structure-constants closure audit now estimates primitive operator centroids for `N,Q,M,T`, projects pairwise commutators into their span, and compares closure residuals to 1000 random-subspace nulls:
  - nonzero overall mean closure residual is below random-subspace null (`0.885` versus `0.984`)
  - strongest Jacobi-like closure triples are `NMT` (`0.309`) and `QMT` (`0.333`)
  - `NQT` remains weak (`0.853`)
  - Chinese `NM-MN` and `MT-TM` rows have zero commutator norm under current templates and must be treated as template degeneracies

Scientific status:

This is promising but more fragile. It should be framed as a null-controlled diagnostic, not as proof of a Lie algebra or as a formal Jacobi identity.

Main risk:

Hand-written templates may induce or suppress cancellation. GPT-2/DistilGPT-2 support `QMT` below-null cancellation, and the grammar-generated pairwise control shows commutator coherence below nulls. The multilingual max audit strengthens the existence of a signed-permutation signal but also revises the story: `QMT` is not uniquely strongest once we move to multilingual templates and multilingual encoders. The structure-constants audit adds partial closure-like evidence, but the Chinese zero-commutator rows show that template degeneracy can create misleadingly clean cells. Endpoint controls remain too strong. The next experiments need endpoint-balanced multilingual generation, target-only third-order composition controls, affine/multiplicative operator maps, and a focused explanation of why different template regimes change the relative ordering of `NQM` and `QMT`.

Important adjacent method:

Linear Relational Decoding suggests that the additive endpoint-delta framing may be too weak for a serious Lie-style claim. The current GLT-MOLT variant now learns relation-specific maps:

```text
y ~= W_op x + b_op
```

and compares additive delta-only, linear matrix-only, and affine map variants. This allows commutators and Jacobi-like residuals to be computed directly over learned operators instead of endpoint sums.

GLT-MOLT pilot result:

The first affine/operator audit is now complete for 7 multilingual models and 7 languages. It compares additive centroid maps, linear matrix maps, and affine `W_op x + b_op` maps for `N,Q,M,T`.

- Additive maps predict one-step targets best (`mean target cosine = 0.895`).
- Linear and affine maps are worse for target reconstruction (`0.735` and `0.693`).
- Matrix commutators show weak but consistent closure-like compression against random-subspace nulls (`linear 0.970` and `affine 0.965` versus random null `~0.9999`).
- Matrix Jacobi-like residuals are very small (`linear 0.067`, `affine 0.063`), but this should be interpreted cautiously because ridge-regularized maps may be algebraically smoother than they are predictively useful.

Current interpretation:

Simple additive displacement vectors outperform learned linear/affine operators at target prediction, while matrix operators are worse predictors but produce cleaner algebraic diagnostics. This split is useful: endpoint differences appear better suited to target movement, while learned operators are useful for asking algebraic closure questions.

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
- A non-leaky Hybrid RISE-Procrustes transfer test now scores each pair against every MDV/RISE-style class prototype and trains transformation-label classifiers on `delta_only`, prototype scores, and `delta + prototype scores`. The hybrid does not improve cross-model label F1: `delta_only` remains `0.684651`, while the best hybrid/prototype variant is `mdv_raw_hybrid_delta_scores` at `0.430839`.
- A movement-level spherical delta steering test now compares linear centroid steering, tangent/exp-map steering, RISE-style prediction, residual orderings, and hybrid averaging. Cross-model target cosine is best for `linear_delta` (`0.613559`), while retrieval label F1 is best for `rise_only` (`0.445518`); `spherical_delta` slightly improves label F1 over `linear_delta` (`0.412695` vs `0.407674`) but lowers target cosine (`0.604795` vs `0.613559`).

Current status:

This is now a serious candidate for a complementary paper, but not as a "first universal geometry" claim. RISE is stronger on cross-lingual/cross-model geometric transformation modeling. Our distinct angle is stress testing: null-controlled Procrustes transfer, held-out anchors, endpoint controls, and comparison against simpler delta/MDV/RISE-style baselines.

Required gates before promotion:

1. Add bootstrap confidence intervals and direction-family summaries for the held-out alignment and RISE-aware comparison curves.
2. Stress-test anchor-domain diversity, e.g. anchors from a different template family or natural paraphrase pool.
3. Improve the RISE-aware comparison:
   - verify the simplified `rise_style` baseline against the published RISE implementation if feasible
   - add confidence intervals
   - clarify target-prediction versus class-discrimination metrics
4. Extend movement-level composition with calibration:
   - learn per-class step sizes for spherical delta steering on train only
   - add bootstrap confidence intervals over directions and examples
   - compare residual orderings against a more faithful published RISE implementation if feasible
5. Reverse-direction transfer summary by model family, e.g. small model to large model and large model to small model.
6. Stronger architectures if feasible, such as Llama/Mistral-class embedding spaces or high-quality sentence encoders.
7. Package the result as a standalone Track 3 draft only after the RISE/MDV comparison and confidence-interval checks.

## Track 4: GLT-STEER, transformation vectors as editors

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

First result:

A focused GPT-2 question activation-steering run is complete.

- Experiment script: `scripts/run_gpt2_activation_steering_pilot.py`
- Result directory: `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`
- Model: `gpt2`
- Transformation evaluated: `question`
- Residual layers: `2, 3, 4, 5, 6`
- Controls: `none`, `target`, `wrong_class`, `random_norm`, `negative_target`
- Raw generations: `6800`
- Failures: none

Main finding:

- At layer `2`, gain `0.75`, target question steering produced question marks in `93.75%` of generations.
- Across all tested layers at gain `0.75`, target question steering produced question marks in `93.50%` of generations.
- Random-norm, wrong-class, and negative-target controls produced `0.00%` question marks in the aggregate control summary.

Reviewer-facing follow-up controls:

- Experiment script: `scripts/run_gpt2_question_steering_controls.py`
- Result directory: `results/experiments/gpt2_question_steering_controls_20260714_results/`
- No-steering question-mark base rate is `0.0000` for both in-template prompts (`160` rows) and out-of-template freeform prompts (`80` rows).
- Under the same compact setting family (`layers 2,3`, gain `0.75`), in-template target steering reaches question-mark rate `0.9625`, while random-norm, wrong-class, and negative-target controls remain at `0.0000`.
- On `40` freeform out-of-template declarative sentences, target steering still reaches question-mark rate `0.8375`, while no-steering and all compact controls remain at `0.0000`.
- A prompt-robustness control then tests four prompt wrappers (`Input/Output`, `Source/Response`, `Sentence/Continuation`, and bare statement prompts). Target steering remains strong across all wrappers:
  - in-template target question-mark rate range: `0.7750-0.9875`
  - out-of-template target question-mark rate range: `0.8125-0.9250`
  - no-steering and wrong-vector controls remain at `0.0000` except one random-norm in-template plain-statement row at `0.0250`
- A content-preservation audit shows that prompt wording determines whether the question marker is added while preserving the source sentence:
  - first-pass target preserved-with-question rates are strong for structured prompts (`0.7500-0.8500` in-template for `Input/Output` and sentence-continuation prompts), but weak for plain out-of-template prompts (`0.0875`) and source-response out-of-template prompts (`0.2125`)
  - copy-oriented prompts improve the result substantially: `repeat_sentence`, `same_sentence`, and `copy_sentence` reach target question-and-preserved rates of `0.9625-0.9750` in-template and `0.8250-0.9000` out-of-template
  - `quoted_echo` remains weak for content preservation despite question-mark generation, so prompt format is a real boundary condition rather than cosmetic wording
- A copy-prompt none-baseline audit directly tests the prompt-only alternative. Across GPT-2 and DistilGPT-2, copy-like prompts without steering produce `0.0000` question-mark rate across `960` no-steering rows.
- A DistilGPT-2 replication is qualitatively positive but much weaker than GPT-2: the best target question-and-preserved rate is `0.4625`, while controls remain at `0.0000` on the joint metric.
- A first non-question extension to negation is weak/negative. Under the same copy-prompt design, negation target rows do not clearly beat matched controls; the best target-and-preserved row is `0.1125`, while the matched control maximum is `0.1375`.

Interpretation:

This is the first behavior-level intervention result in GLT. It supports the cautious claim that a question-transformation activation vector can steer GPT-2 toward question-like output form under residual-stream injection. The copy-prompt follow-up strengthens the result from mere punctuation steering toward partial form-preserving rewriting, but only under prompt families that already ask the model to repeat or copy the source sentence. The no-steering copy-prompt audit rules out the simplest prompt-only explanation for question marks, while the DistilGPT-2 replication shows that effect size is model-dependent. The negation attempt shows that the steering recipe does not automatically transfer to other transformations.

Caveats:

- This is currently a GPT-2-only, question-only result.
- It shows output-form steering plus prompt-dependent content preservation, not robust general-purpose semantic rewriting.
- The out-of-template control is promising but modest: `40` hand-written freeform declarative sentences.
- The prompt-robustness result reduces concern about one prompt wrapper, but it still uses simple declarative prompts and should be extended to more natural contexts.
- The content-preservation result depends strongly on prompt wording; copy-like prompts are much cleaner than quoted or bare prompts.
- DistilGPT-2 weakens the effect substantially, so the result is not architecture-invariant.
- Negation steering fails under the current copy-prompt design, so transformation-class dependence is now explicit.
- The broad pilot showed weak or noisy results for negation, modality, and tense shift, so other transformations need redesigned prompts, metrics, or steering sites.
- The result connects GLT to representation steering, but it is not evidence for a complete linguistic algebra.

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

2026-06-23 status:

The first large multilingual Track 2 audit is complete for 7 languages and 5 multilingual encoders. It is not yet a clean cross-lingual universality result, because the templates are synthetic and endpoint controls remain strong. Still, it is the strongest current evidence that the signed-permutation diagnostic is not purely an English-only artifact.

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

## Track 7: GLT-AFFECT, graded affective transformation geometry

Working title:

**GLT-AFFECT: Graded Affective Transformation Geometry**

Central hypothesis:

Some semantic transformations are not binary switches but graded axes. Emotional polarity is the first controlled case:

```text
hate (-2) -> dislike (-1) -> indifferent (0) -> like (+1) -> love (+2)
```

This turns the project from class separability toward measuring the shape of a semantic axis:

- linearity: does `indifferent -> like` plus `like -> love` approximate `indifferent -> love`?
- curvature: are adjacent step norms and directions stable across the scale?
- saturation: are extreme steps compressed or expanded relative to middle steps?
- opposition: is `neutral -> love` approximately opposite to `neutral -> hate`?

Methodological guardrails:

- Antisymmetry of `delta(A -> B) = -delta(B -> A)` is a subtraction identity and should only be used as a sanity check.
- Affect/negation examples must not be treated as commutators until both orders are defined as paths to comparable endpoints. "I do not love you" and "I hate you" are different semantic states, not interchangeable endpoints.
- Text-only models can test the geometry of language about affective states, not the lived affective or sensory reality itself.
- Grounding claims require independent non-textual anchors, such as psychophysical odor datasets.

MVP:

1. Build a text-only graded affect dataset over one polarity scale: hate, dislike, indifferent, like, love.
2. Use multiple subjects, objects, and contexts while keeping lexical leakage explicit and auditable.
3. Compute adjacent and non-adjacent delta vectors.
4. Test linearity, curvature, saturation, and opposition under model/language variation.
5. Report the result as language-representation geometry, not real-world affect grounding.

MVP result:

The first GLT-AFFECT polarity run is complete for 7 multilingual models and 7 languages.

- Adjacent affective step norms are not uniform:
  - `hate -> dislike`: `3.90`
  - `dislike -> indifferent`: `5.05`
  - `indifferent -> like`: `5.38`
  - `like -> love`: `3.31`
- The coefficient of variation across adjacent step norms is `0.224`, suggesting a non-uniform scale.
- `neutral -> love` and `neutral -> hate` are not opposite directions; their mean row cosine is positive (`0.614`) with centroid cosine `0.613`.
- The safe interpretation is that text embeddings represent love and hate as movements away from indifference into emotionally loaded language regions, not as simple antipodal valence directions.
- The path-additivity sanity check is numerically exact, as expected from vector subtraction, and should not be presented as evidence.

Marker-pooling control:

The first artifact check replaces sentence-level mean pooling with marker-only pooling over the affective lexical marker span. This tests whether the love/hate same-direction result is merely caused by averaging the shared sentence template.

- Experiment script: `scripts/run_glt_affect_marker_pool_control.py`
- Result directory: `results/experiments/glt_affect_marker_pool_control_results/`
- Models: 7 multilingual embedding/backbone models.
- Languages: `en`, `ru`, `zh`.
- Templates per language: `160`.
- Marker span recovery: `1.000` for every language/level cell.
- `neutral -> love` versus `neutral -> hate` mean row cosine:
  - sentence mean pooling: `0.590`
  - marker-only pooling: `0.526`
- Adjacent norm coefficient of variation:
  - sentence mean pooling: `0.238`
  - marker-only pooling: `0.203`

Interpretation:

The effect weakens under marker-only pooling but does not disappear. Therefore the first GLT-AFFECT result is not fully explained by sentence-template mean-pooling artifacts. The conservative claim is that a same-direction affective-involvement component survives at the lexical marker level. This remains a text-embedding result, not a claim about real affective experience.

Next control:

Compare the affective scale against non-affective marker scales and random lexical marker ladders under the same pooling pipeline. If neutral/random marker scales show the same positive endpoint geometry, the GLT-AFFECT interpretation must be weakened to a generic lexical-substitution geometry. If affect remains distinct, the affective-involvement interpretation becomes stronger.

Lexical-specificity control:

The second artifact check compares the affective scale against size, attention, and random-label marker ladders using the same templates, languages, models, and marker-pooling pipeline.

- Experiment script: `scripts/run_glt_affect_lexical_specificity_control.py`
- Result directory: `results/experiments/glt_affect_lexical_specificity_control_results/`
- Models: 7 multilingual embedding/backbone models.
- Languages: `en`, `ru`, `zh`.
- Scales: `affect`, `size`, `attention`, `random_label`.
- Templates per scale/language: `120`.
- Marker span recovery: `1.000` for every scale/language/level cell.
- `neutral -> +2` versus `neutral -> -2` mean row cosine:
  - marker-only pooling:
    - `affect`: `0.524`
    - `random_label`: `0.486`
    - `size`: `0.421`
    - `attention`: `0.418`
  - sentence mean pooling:
    - `affect`: `0.590`
    - `random_label`: `0.504`
    - `size`: `0.396`
    - `attention`: `0.361`

Interpretation:

The affect scale is the strongest of the tested ladders, but the random-label ladder is also substantially positive. Therefore the current claim should not be phrased as a uniquely emotional geometry. A safer formulation is that affect shows an excess same-direction endpoint geometry over several lexical controls, while a nontrivial part of the effect is shared with generic lexical replacement.

Immediate statistical follow-up:

Compute paired bootstrap confidence intervals for `affect - random_label`, `affect - size`, and `affect - attention` over matched model/language/template cells. This decides whether the affect-leading result is statistically stable rather than a mean-table artifact.

Bootstrap contrast result:

The paired bootstrap follow-up is complete over matched model/language/template cells with `20,000` bootstrap resamples.

- Experiment script: `scripts/run_glt_affect_lexical_contrast_bootstrap.py`
- Result directory: `results/experiments/glt_affect_lexical_contrast_bootstrap_results/`
- Input rows: `20,160`.
- Contrast rows: `15,120`.
- Global marker-only contrasts:
  - `affect - random_label`: mean `0.038`, 95% CI `[0.031, 0.045]`
  - `affect - size`: mean `0.104`, 95% CI `[0.095, 0.112]`
  - `affect - attention`: mean `0.106`, 95% CI `[0.098, 0.114]`
- Global sentence mean-pooling contrasts:
  - `affect - random_label`: mean `0.086`, 95% CI `[0.076, 0.095]`
  - `affect - size`: mean `0.194`, 95% CI `[0.184, 0.204]`
  - `affect - attention`: mean `0.229`, 95% CI `[0.220, 0.238]`

Interpretation:

The contrast result supports a small but stable affect-specific excess over generic random-label replacement under marker-only pooling. The excess is much smaller than the raw affect signal, so GLT-AFFECT should be framed as an affect-leading lexical geometry result with a substantial generic lexical-substitution component.

Current interpretation:

GLT-AFFECT is promising because it exposes a graded, curved affective geometry rather than another categorical classifier task. The marker-only control makes the first signal harder to dismiss, but the program still needs lexical-specificity controls before it can be promoted beyond cautious language-representation evidence.

Longer-term grounding track:

Use Pyrfume and related psychophysical odor datasets to compare text-derived descriptor geometry against independent perceptual dissimilarity matrices. This is a separate GLT-AFFECT-GROUNDING subtrack, not part of the first text-only MVP.

Scientific payoff:

GLT-AFFECT introduces the first graded semantic axis in the project. If it works, it gives a natural way to study deformation, curvature, and saturation in embedding space without immediately relying on binary operator composition.

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
2. Keep UPAT as a bounded hard-holdout result unless future matched-capacity tests change it:
   - either expand UPAT and match train sizes against the main dataset
   - or keep it explicitly as a bounded hard-holdout result and narrow Track 1 claims
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
11. Treat endpoint-subspace residualization as complete for the current linear-control layer:
   - sign-direction residualization is complete
   - endpoint-position subspace removal is complete
   - triple-label subspace removal is complete
   - joint sign/triple/position removal is complete
12. Add target-only and endpoint-only baselines for third-order multilingual signed-permutation endpoints.
13. Add nonlinear endpoint-artifact controls, such as adversarial endpoint balancing or kernel/MLP endpoint probes.
14. Extend the completed GLT-MOLT matched-null and spectral-null results:
   - optionally complete a separate PCA-256 spectral-null sensitivity job after the compact PCA-64 and PCA-128 results
   - test whether the operator-closure signal survives across layers
   - add a second spectral-null pass at multiple ridge alphas if the `alpha=100` result becomes central to the paper
15. Add the GLT-AFFECT text-only MVP:
   - graded emotional polarity scale
   - linearity and curvature tests
   - no affect/negation commutator claim until endpoints are formally controlled
16. Build an endpoint-balanced multilingual generator and re-run the 7-language audit.
17. Explain the `NQM` versus `QMT` reversal between the English/decoder table and the multilingual max audit.
18. Regenerate a dated PDF packet after every major run.

### Medium term

1. Extend GLT-STEER after the focused question result:
   - repeat the question steering result with a second seed or prompt family
   - add a stricter semantic-preservation metric
   - test whether prompt cleanup reduces repetition
   - rerun negation, modality, and tense shift with transformation-specific metrics
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
   - multiseed standard deviations and seed-level effect intervals are reported
   - at least one prior-work baseline or comparison is written up, such as task vectors or function vectors
   - at least one model outside the original five is included as a spot-check; currently satisfied by BERT-large and DeBERTa-v3-base
   - UPAT hard-holdout is either resolved experimentally or explicitly reported as a bounded hard-holdout result
2. Keep Track 2 as a diagnostics paper until grammar-generated templates and endpoint-only controls succeed.
3. Promote Track 3 only if it becomes clearly complementary to RISE: null-controlled Procrustes transfer plus held-out anchors plus an explicit RISE/MDV comparison. The main narrative should be stress-testing cross-model transfer, not claiming first discovery of universal transformation geometry.
4. If GLT-STEER replicates beyond question formation, write Track 4 as an intervention/controllable-generation paper.
5. If cross-lingual transfer works, it becomes the strongest version of the universality claim.
6. If Track 2 survives grammar-generated controls, write it as a separate diagnostics paper rather than merging it into Track 1.
7. If Track 2 weakens under controls, keep it as a negative/diagnostic section in a broader research note.
