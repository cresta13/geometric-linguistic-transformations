# Third-Order Signed Permutation Coherence for Linguistic Transformations in Transformer Embedding Spaces

## Abstract

If linguistic transformations are recoverable as displacement vectors, the next question is whether they compose in structured ways. We test controlled transformations corresponding to negation (`N`), question formation (`Q`), modality/evidentiality (`M`), and tense (`T`). Ordered compositions differ in embedding space, and semantic-equivalence controls show statistically significant but distributionally broad separation between equivalent and non-equivalent pairs. For third-order compositions, we no longer describe the main test as a Jacobi identity. Instead, we evaluate a neutral diagnostic: third-order signed permutation coherence, defined by the alternating endpoint sum `ABC+BCA+CAB-ACB-CBA-BAC` and compared against permutation-null baselines with bootstrap intervals. The strongest current result is local: `QMT` shows robust below-null cancellation across BERT, DistilRoBERTa, and RoBERTa, while negation-containing triples are mixed or worse than null. The evidence supports a local signed-permutation coherence effect, not a global Lie algebra.

## Evidence Map

Main scripts:

- Composition audit: [run_lie_composition_audit.py](../../../scripts/run_lie_composition_audit.py)
- Semantic equivalence control: [run_lie_semantic_equivalence_control.py](../../../scripts/run_lie_semantic_equivalence_control.py)
- Algebraic-identity audit / signed permutation diagnostic: [run_lie_algebraic_identities.py](../../../scripts/run_lie_algebraic_identities.py)
- Composition dataset helper: [lie_composition_dataset.py](../../../scripts/lie_composition_dataset.py)

Main result files:

- Composition summary: [lie_composition_summary.csv](../../../results/experiments/lie_composition_results/csv/lie_composition_summary.csv)
- Decoder composition summary: [lie_composition_summary.csv](../../../results/experiments/lie_composition_decoder_results/csv/lie_composition_summary.csv)
- Semantic equivalence summary: [semantic_equivalence_summary.csv](../../../results/experiments/lie_semantic_equivalence_results/csv/semantic_equivalence_summary.csv)
- Semantic statistical tests: [semantic_equivalence_tests.csv](../../../results/experiments/lie_semantic_equivalence_results/csv/semantic_equivalence_tests.csv)
- Semantic effect sizes: [reviewer_semantic_effect_sizes.csv](../../../results/experiments/lie_semantic_equivalence_results/csv/reviewer_semantic_effect_sizes.csv)
- Signed permutation summary: [jacobi_summary.csv](../../../results/experiments/lie_algebraic_identities_results/csv/jacobi_summary.csv)
- Signed permutation raw rows: [jacobi_raw_all_models.csv](../../../results/experiments/lie_algebraic_identities_results/csv/jacobi_raw_all_models.csv)
- Multiple-testing correction: [signed_permutation_multiple_testing.csv](../../../results/experiments/lie_algebraic_identities_results/csv/signed_permutation_multiple_testing.csv)
- Dataset audit: [reviewer_dataset_audit.csv](../../../results/experiments/lie_algebraic_identities_results/csv/reviewer_dataset_audit.csv)
- Decoder signed permutation summary: [jacobi_summary.csv](../../../results/experiments/lie_algebraic_identities_decoder_results/csv/jacobi_summary.csv)

Daily verification packet:

- [2026-06-12_reviewer_revised_report.pdf](../../../reports/2026-06-12_reviewer_revised_report.pdf)

## 1. Motivation

The geometric transformation-vector paper shows that delta vectors add information beyond endpoints. This paper asks a stricter question:

> Do linguistic transformations compose in a way that reveals structured operator-like behavior?

The answer is currently partial. Pairwise composition shows order effects. Third-order signed permutation tests show one robust local effect (`QMT`) and several failures involving negation.

## 2. Related Work Positioning

This track is related to task arithmetic and function-vector work, but it asks a different question. Task arithmetic composes vectors in model weight space after fine-tuning; function vectors identify activation directions that can causally induce in-context functions. Here the objects are not weight updates or causal activation interventions, but ordered sentence-composition endpoints in embedding space.

The closest conceptual overlap is relation-vector probing and representation steering: we also ask whether semantic or linguistic transformations produce reusable directions. The difference is that this paper tests ordered composition explicitly (`AB` versus `BA`) and then tests a null-controlled signed third-order endpoint sum.

Negation literature is also central. Prior negation probes show that pretrained language models can fail to robustly distinguish negated from non-negated factual prompts. In our setting, Track 1 shows that negation can be easy as a one-step surface-labeled class, while Track 2 shows that negation is unstable inside ordered third-order composition. That distinction is one of the main reasons to keep the two paper tracks separate.

Working references:

- Freenor and Alvarez 2026, "Mapping Semantic & Syntactic Relationships with Geometric Rotation" (RISE).
- Ilharco et al. 2023, "Editing Models with Task Arithmetic".
- Todd et al. 2024, "Function Vectors in Large Language Models".
- Park, Choe, and Veitch 2023/2024, "The Linear Representation Hypothesis and the Geometry of Large Language Models".
- De Raedt et al. 2021, "A Simple Geometric Method for Cross-Lingual Linguistic Transformations with Pre-trained Autoencoders".
- Kassner and Schuetze 2020, "Negated and Misprimed Probes for Pretrained Language Models".
- Huang et al. 2024, "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations".

Closest-work distinction:

RISE is the closest methodological neighbor, but this track asks a different question. RISE estimates spherical/geodesic rotations that map semantic-syntactic transformations across languages and models, and it frames its learned transformations as commutative in tangent-space composition. This draft does not propose a steering or target-embedding prediction method. It asks whether ordered transformation endpoints show local composition structure, especially via pairwise noncommutativity and third-order signed-permutation cancellation. The strongest current result is deliberately narrow: `QMT` is coherent under this diagnostic, while negation-containing triples are unstable.

This distinction is now central rather than cosmetic. RISE strengthens the case that some transformations can be treated as reusable geometric operations; our Track 2 asks where ordered linguistic composition deviates from simple commutative addition. If this result survives grammar-generated templates and endpoint-only controls, it becomes a complementary diagnostics paper: RISE-style methods describe transferable one-step operations, while this track probes order sensitivity and local composition failures.

Compared with LRH work, the signed-permutation diagnostic should not be read as a formal proof of a linear representation theorem. Compared with De Raedt et al. 2021, this is not a multilingual autoencoder property-transfer method; it is a controlled transformer-embedding diagnostic over ordered English sentence transformations.

## 3. Operations and Models

Current operations:

| Symbol | Operation |
|---|---|
| `N` | negation |
| `Q` | question formation |
| `M` | modality/evidentiality via allegedness |
| `T` | tense/future temporal shift |

Current encoder models:

- `bert-base-uncased`
- `distilroberta-base`
- `roberta-base`

Decoder replication now added:

- `gpt2`
- `distilgpt2`

Reviewer-critical caveat:

The decoder replication supports the local `QMT` result, but negation-containing triples become more model-dependent. This strengthens the paper's narrowed framing: local signed-permutation coherence exists, but it is not a general operator algebra.

## 4. Composition and Noncommutativity

The first test compares ordered compositions `AB` and `BA`.

![Composition noncommutativity heatmap](../../../results/experiments/lie_composition_results/figures/01_noncommutativity_heatmap.png)

**Figure 1.** Noncommutativity heatmap for ordered operation pairs.

![Relative commutator norm heatmap](../../../results/experiments/lie_composition_results/figures/02_relative_commutator_norm_heatmap.png)

**Figure 2.** Relative commutator norm across operation pairs and models.

Observation:

Order matters, especially for transformations involving tense. This is necessary but not sufficient evidence for operator structure, because template wording alone could create order-sensitive embeddings.

Decoder composition has now been added, so decoder models are no longer present only in the favorable third-order diagnostic.

![Decoder composition noncommutativity heatmap](../../../results/experiments/lie_composition_decoder_results/figures/01_noncommutativity_heatmap.png)

**Figure 3.** Decoder-model pairwise noncommutativity heatmap.

Key decoder rows:

| Model | Pair | Mean noncommutativity |
|---|---|---:|
| DistilGPT-2 | `NT_vs_TN` | `0.295` |
| DistilGPT-2 | `MT_vs_TM` | `0.289` |
| DistilGPT-2 | `QM_vs_MQ` | `0.196` |
| GPT-2 | `QM_vs_MQ` | `0.111` |
| GPT-2 | `NQ_vs_QN` | `0.066` |

Decoder pairwise composition is weaker for GPT-2 than for DistilGPT-2 under this mean-pooled PCA diagnostic, but it is not absent. This asymmetry should be reported rather than hidden.

## 5. Semantic Equivalence Control

The semantic equivalence control compares pairs intended to preserve meaning against pairs where order should change meaning.

![Semantic equivalence control](../../../results/experiments/lie_semantic_equivalence_results/figures/01_equivalent_vs_nonequivalent.png)

**Figure 4.** Equivalent pairs have lower mean noncommutativity, but distributions are broad.

Summary and tests:

| Model | Equivalent mean | Non-equivalent mean | Difference | Cohen d | Welch p | Mann-Whitney p |
|---|---:|---:|---:|---:|---:|---:|
| BERT | `0.113` | `0.329` | `0.216` | `2.53` | `2.21e-140` | `4.92e-131` |
| DistilRoBERTa | `0.129` | `0.232` | `0.102` | `2.35` | `2.17e-150` | `4.96e-117` |
| RoBERTa | `0.154` | `0.278` | `0.124` | `2.29` | `9.19e-142` | `5.07e-103` |

Interpretation:

The difference is statistically strong. But the control should not be oversold as a clean separation: the standard deviations are substantial and distributions overlap. The correct claim is that semantic-equivalence labels shift the distribution of noncommutativity, not that they perfectly separate pair types.

## 6. Tautological Antisymmetry Check

The implementation also checks whether:

```text
[A,B] ~= -[B,A]
```

This is not a scientific result. Given the endpoint definition:

```text
[A,B] = delta_AB - delta_BA
[B,A] = delta_BA - delta_AB
```

the identity `[A,B] = -[B,A]` follows algebraically for any vectors. We therefore do not treat perfect antisymmetry as evidence about transformers. The check is kept only as an implementation sanity test: if it failed, the code would be wrong.

## 7. Third-Order Signed Permutation Coherence

We avoid calling the main third-order test a Jacobi identity. The Lie-algebra Jacobi identity is:

```text
[A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0
```

Our endpoint-based diagnostic is different:

```text
S(A,B,C) = ABC + BCA + CAB - ACB - CBA - BAC
```

This is a signed alternating sum over six composition endpoints. It tests whether cyclic orderings and anti-cyclic orderings occupy a coherent signed geometry in embedding space. It is related in spirit to antisymmetric third-order structure, but it is not a formal Jacobi identity.

Metrics:

```text
relative_signed_permutation_norm = ||S|| / scale
signed_permutation_to_null_ratio = observed relative norm / permutation-null mean
```

The CSV still uses historical column names such as `relative_jacobi_norm` and `jacobi_to_null_mean_ratio`; these should be renamed in code before submission.

Controls:

- 1000 permutation-null samples per row
- 2000 bootstrap resamples for confidence intervals
- all four triples from `N,Q,M,T`: `NQM`, `NQT`, `NMT`, `QMT`
- endpoint audit: no duplicate endpoints in any row

Dataset audit:

| Triple | Rows | Unique sources | Unique endpoint texts | Duplicate endpoint rows |
|---|---:|---:|---:|---:|
| `NMT` | 100 | 100 | 600 | 0 |
| `NQM` | 100 | 100 | 600 | 0 |
| `NQT` | 100 | 100 | 600 | 0 |
| `QMT` | 100 | 100 | 600 | 0 |

![Signed permutation relative norm heatmap](../../../results/experiments/lie_algebraic_identities_results/figures/03_jacobi_relative_norm_heatmap.png)

**Figure 5.** Relative norm of the third-order signed permutation sum.

![Signed permutation versus null](../../../results/experiments/lie_algebraic_identities_results/figures/04_jacobi_vs_permutation_null_heatmap.png)

**Figure 6.** Ratio of observed signed permutation norm to permutation-null mean. Values below `1.0` indicate stronger-than-null cancellation.

![Decoder signed permutation ratio](../../figures/decoder_signed_permutation_ratio.png)

**Figure 7.** Decoder-model replication for the signed permutation ratio to permutation-null mean.

## 8. Main Third-Order Result

The most robust cross-architecture below-null result is `QMT`.

| Model | Triple | Ratio to null | 95% CI |
|---|---|---:|---:|
| BERT | `QMT` | `0.683` | `[0.677, 0.689]` |
| DistilRoBERTa | `QMT` | `0.631` | `[0.627, 0.636]` |
| RoBERTa | `QMT` | `0.623` | `[0.617, 0.629]` |
| GPT-2 | `QMT` | `0.539` | `[0.519, 0.561]` |
| DistilGPT-2 | `QMT` | `0.771` | `[0.745, 0.797]` |

Negative or mixed cases:

| Model | Triple | Ratio to null | Interpretation |
|---|---|---:|---|
| BERT | `NQT` | `1.117` | worse than null |
| DistilRoBERTa | `NMT` | `1.134` | worse than null |
| RoBERTa | `NMT` | `1.253` | worse than null |
| RoBERTa | `NQM` | `0.980` | near null |
| GPT-2 | `NMT` | `1.335` | worse than null |
| GPT-2 | `NQM` | `1.195` | worse than null |
| DistilGPT-2 | `NQT` | `1.420` | worse than null |

Multiple-testing correction:

We applied a row-bootstrap test of whether the mean signed-permutation ratio is below `1.0`, followed by Bonferroni and Benjamini-Hochberg correction over all `4 triples x 5 models = 20` model/triple tests.

| Triple | Models passing Bonferroni below-null | Interpretation |
|---|---:|---|
| `QMT` | `5/5` | only fully cross-architecture stable below-null triple |
| `NQM` | `3/5` | below-null in BERT, RoBERTa, DistilGPT-2; fails in DistilRoBERTa and GPT-2 |
| `NMT` | `2/5` | below-null in BERT and DistilGPT-2; worse than null in several models |
| `NQT` | `1/5` | below-null only in GPT-2; worse than null elsewhere |

Therefore, the central claim is now narrower:

> `QMT` is the only tested triple with stable third-order signed permutation cancellation stronger than null across all five tested models after table-level multiple-testing correction.

Working hypothesis for why `QMT` is coherent:

`Q`, `M`, and `T` are all clause-level operators that modify illocution, epistemic status, or temporal anchoring while preserving the same event frame. Their endpoints can remain relatively aligned around one proposition. Negation (`N`) changes truth-conditional polarity and often introduces lexical/scope markers that interact with syntax more sharply. This makes `N` easy to detect as a one-step surface-labeled transformation, but less stable as a component in ordered composition.

This is still a hypothesis, not a proven linguistic theory. It gives a pre-registered direction for the next template-generator experiment: if QMT coherence is real, it should survive grammar-generated paraphrases where question, modality, and tense vary without introducing negation.

Critical template caveat:

The original Lie-style dataset is synthetic and hand-written. Several operations have stable lexical markers: negation often introduces "failed to", questions often begin with a fixed auxiliary pattern, and tense can introduce explicit temporal markers. Variant transfer partially reduces this concern in Track 1, but the Lie-style composition experiments still need endpoint-balanced grammar generation before submission. Until then, the Track 2 claim is a diagnostic result about controlled probe sets, not a general claim about natural-language operator algebra.

Update after the first grammar-generated pairwise control:

- A grammar-generated `N,Q,M,T` pairwise composition probe now exists.
- Observed relative commutator norms remain below `N=1000` shuffled and norm-matched nulls.
- Endpoint-only and delta-only controls classify pair labels almost perfectly.

This improves the evidence that pairwise order structure is not destroyed by grammar variation, but it does not solve the endpoint-artifact problem. The next version needs endpoint-balanced grammar generation and target-only controls for third-order composition endpoints.

## 9. Negation Is a Result, Not Just a Limitation

Negation-containing triples are where the structure breaks most clearly. This should be interpreted as a substantive finding.

Working hypothesis:

Negation may not behave like a smooth linear direction in these embedding spaces. It can interact with scope, syntax, and surface markers in ways that make endpoint geometry less coherent. This aligns with the broader concern that transformer representations often handle negation less robustly than positive factual content.

Immediate follow-up:

- isolate negation-only transformations in Track 1 confusion analysis
- compare `N` pairwise commutators against non-`N` pairwise commutators
- generate paraphrases where negation markers are less lexically obvious
- test whether middle layers represent negation more coherently than final layers

Decoder replication sharpens this point. GPT-2 shows strong `QMT` cancellation (`0.539`) but fails on `NMT` and `NQM`; DistilGPT-2 keeps `NMT` and `NQM` below null but fails badly on `NQT`. The important fact is not merely that negation is hard, but that negation changes the algebraic diagnostic differently across architectures.

## 10. What This Evidence Does and Does Not Prove

Evidence layers:

1. Pairwise composition gives noncommutativity.
2. Semantic controls show a statistically significant distribution shift.
3. Signed permutation coherence identifies one robust local third-order effect.
4. Multiple-testing correction shows that QMT is the only below-null triple passing across all five models.
5. Negation triples are model-dependent, which constrains the theory.
6. Grammar-generated pairwise controls preserve below-null commutator coherence, but endpoint controls remain too strong.

Not proven:

- global Lie algebra
- formal Jacobi identity
- model-independent operator calculus
- endpoint-balanced robustness to grammar-generated templates
- broad decoder-model generality beyond the two spot-checks
- lexical-marker independence of the hand-written composition templates

## 11. Required Pre-Submission Experiments

1. Rename code/CSV columns away from `jacobi_*` toward `signed_permutation_*`.
2. Build endpoint-balanced grammar templates.
3. Add target-only controls for third-order composition endpoints.
4. Add layerwise analysis.
5. Add pooling ablation.
6. Add a focused negation analysis before adding new operators.
7. Add prior-work positioning and bibliography.

## Current Status

This is a promising diagnostic study, but it is not submission-ready. The current result is useful precisely because it is local and falsifiable: `QMT` survives current controls, while negation-heavy triples expose the boundary of the phenomenon.
