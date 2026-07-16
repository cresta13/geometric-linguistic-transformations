# Geometric Linguistic Transformations

Research code, result tables, figures, and draft papers for **GLT** (**Geometric Linguistic Transformations**): a research program for testing whether linguistic transformations appear as reusable geometric objects in transformer embedding spaces.

This is an active and reproducible research program. The Build Week submission is not presented as a finished theorem or a final paper, but as a working experimental pipeline, an evidence package, and an open scientific question.

## OpenAI Build Week 2026 Note

This repository is being used as a Build Week submission in the **Developer Tools** category: the working artifact is a reproducible research workflow rather than a separate web app.

Codex was used as the primary implementation environment for recent scripts, controls, result summaries, documentation updates, and repository organization. GPT-5.6 was used as a reasoning partner for revisiting mathematical background, reading related work, challenging interpretations, and turning vague research questions into concrete controls.

The human role remained to choose the research direction, decide which claims were honest enough to keep, interpret results, and stay responsible for remaining mistakes. The broader project message is that AI can make careful independent scientific work more accessible without removing the need for evidence, criticism, and humility.

Work completed during Build Week is tracked in the public research record rather than in separate submission-only files. The main completed addition is **GLT-STEER**: GPT-2 question activation steering with base-rate controls, out-of-template controls, prompt-robustness checks, and the final content-preservation / copy-prompt follow-up. The latest preservation result is summarized under `GLT-STEER` below and archived in:

- `results/experiments/gpt2_question_content_preservation_20260716_results/`
- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/`

## 1. Plain-Language Overview

Modern language models turn sentences into high-dimensional vectors. This project asks a simple question:

> If one sentence is changed into another sentence, does the vector movement between them describe the kind of linguistic change that happened?

For example:

```text
source sentence -> transformed sentence
statement       -> question
present tense   -> future tense
plain claim     -> uncertain claim
positive claim  -> negated claim
```

For each pair, we compute:

```text
delta = embedding(transformed sentence) - embedding(source sentence)
```

Then we test whether these deltas behave like meaningful transformation objects rather than arbitrary differences between two sentences.

### What We Have Found So Far

The current evidence supports a cautious version of the idea:

- Sentence-pair deltas often contain information about the **type of transformation**.
- In the main full-semantic experiments, deltas beat target-only baselines across several models under Linear SVC probes.
- In harder sentence-pair holdouts such as UPAT, endpoint features can dominate. This suggests that delta geometry captures transformation type better than absolute transformation identity.
- Some ordered transformations show structured composition effects, but this is not a proof of a Lie algebra.
- Simple additive deltas predict target embeddings better than learned linear/affine maps, while learned maps are useful for algebraic closure diagnostics.
- A first affective-scale experiment suggests that text embeddings treat love/hate-like polarity as curved affective-involvement geometry rather than a simple opposite-axis scale.

### What Is Not Claimed

This repository does not claim:

- that transformer embeddings contain a complete linguistic algebra;
- that the reported effects are independent of endpoint wording;
- that syntax-holdout `1.0` results prove deep generalization;
- that antisymmetry checks are scientific evidence;
- that text-only affect geometry is grounded emotional experience;
- that this is a submission-ready paper.

The project is deliberately conservative: positive results, failures, endpoint leakage, and hard-holdout boundaries are all kept in the record.

## 2. Quick Demo

The fastest demo is the GLT-STEER question experiment.

It shows a simple behavior-level result:

```text
Without steering: GPT-2 repeats a statement.
With a question-transformation vector: GPT-2 starts producing question marks.
With random or wrong-class vectors: the question-mark effect does not appear.
```

Run the lightweight demo from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

This command does not download models or rerun the experiment. It reads the archived CSV in:

```text
results/experiments/gpt2_question_activation_steering_focused_20260714_results/
```

Headline result:

| condition | question mark rate |
|---|---:|
| target question vector | `0.9350` |
| random-norm control | `0.0000` |
| wrong-class control | `0.0000` |
| negative-target control | `0.0000` |

Stronger copy-prompt preservation result:

| condition | question-and-preserved rate |
|---|---:|
| target question vector, copy-like prompts | up to `0.9750` |
| wrong-vector / no-steering controls | `0.0000` in the matched headline rows |

Reviewer-facing prompt-only control:

| condition | question mark rate |
|---|---:|
| copy-like prompts without steering, GPT-2 + DistilGPT-2 | `0.0000` across `960` no-steering rows |

Full notes:

- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md`
- `results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md`
- `results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md`
- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md`
- `results/experiments/distilgpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md`
- `results/experiments/question_copy_prompt_none_baseline_20260716_results/SUMMARY.md`
- `results/experiments/gpt2_steering_delta_coherence_20260716_results/SUMMARY.md`

Safe interpretation:

> This is evidence that a question-transformation activation vector can steer GPT-2 toward question-like output form. It is not evidence that semantic editing is solved, and it is not proof of a complete linguistic algebra.

## 3. Technical Overview

GLT is organized into several research tracks.

### GLT-DV: Delta-Vector Diagnostics

Question:

> Do sentence-pair displacement vectors encode linguistic transformation type beyond source-only or target-only endpoint features?

Main evidence:

- Multiseed full-semantic ablations show reproducible `delta > y_only` under Linear SVC across the original model set.
- McNemar tests and seed-level intervals support the Linear SVC delta advantage.
- Large/modern spot-checks include BERT-large and DeBERTa-v3-base.
- Syntax-holdout `1.0` is treated as target/surface leakage because `y_only` also solves it.
- UPAT is kept as a bounded hard-holdout result: endpoint features dominate in that regime.

Main draft:

- `paper/articles/geometric-transformation-vectors/draft.md`

Important result files:

- `results/ablation_multiseed_aggregated.csv`
- `results/ablation_multiseed_mcnemar.csv`
- `results/track1_multiseed_effect_intervals.csv`
- `results/experiments/upat_audit_results/`
- `results/experiments/track1_spotcheck_large_results/`

### GLT-SPOT: Signed-Permutation Operator Tests

Question:

> Do ordered linguistic transformations show nontrivial composition structure?

This track studies operations such as:

| Symbol | Operation |
|---|---|
| `N` | negation |
| `Q` | question formation |
| `M` | modality/evidentiality |
| `T` | tense / temporal shift |

The key diagnostic is a third-order signed endpoint sum:

```text
S(A,B,C) = ABC + BCA + CAB - ACB - CBA - BAC
```

This is not called a Jacobi identity. It is a signed-permutation coherence test compared against null baselines.

Main evidence:

- Pairwise order matters for several transformation pairs.
- Semantic-equivalence controls shift noncommutativity distributions in the expected direction.
- Multilingual audits show below-null signed-permutation ratios across 7 languages and multiple multilingual encoders.
- Endpoint-subspace residualization shows the signal survives removal of several linear endpoint-derived probe subspaces.
- Endpoint controls remain strong, so the result is framed as a controlled diagnostic rather than endpoint-independent algebra.

Main draft:

- `paper/articles/lie-style-linguistic-operators/draft.md`

Important result folders:

- `results/experiments/lie_multilingual_max_results/`
- `results/experiments/lie_endpoint_subspace_9m_96t_pca128_results/`
- `results/experiments/lie_multilingual_triple_endpoint_controls_7m_96t_results/`
- `results/experiments/lie_structure_constants_results/`

### GLT-MOLT: Matrix/Operator Diagnostics

Question:

> If transformations are learned as linear or affine maps, do their commutators show closure-like structure?

This track compares:

```text
additive: y ~= x + delta_op
linear:   y ~= W_op x
affine:   y ~= W_op x + b_op
```

Main evidence:

- Simple additive displacement vectors outperform learned linear/affine operators at target prediction.
- Learned operators are worse target predictors but expose weak matrix-commutator closure under random-subspace, norm-matched, signed-permutation, and spectral null controls.
- Ridge sweeps show that algebraic cleanliness improves under stronger regularization, so closure results are reported as controlled compression diagnostics, not as a formal Lie-algebra proof.
- Compact PCA-64 and PCA-128 sensitivity checks preserve the spectral-null closure signal on five stable multilingual encoders.

Important result folders:

- `results/experiments/glt_molt_affine_operator_9m_160t_1000null_results/`
- `results/experiments/glt_molt_ridge_sweep_9m_160t_300null_results/`
- `results/experiments/glt_molt_matched_nulls_9m_160t_a10_100_1000null_results/`
- `results/experiments/glt_molt_spectral_nulls_9m_160t_a100_300null_g256_results/`
- `results/experiments/glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256_results/`
- `results/experiments/glt_molt_spectral_pca128_5m_160t_a100_300null_g256_results/`

### GLT-XFER: Cross-Model Transfer

Question:

> Does transformation geometry transfer across embedding models after alignment?

Main evidence:

- UPAT-large Procrustes transfer survives `N=1000` random-label, random-pairing, and random-orthogonal null controls.
- Held-out anchor alignment-size curves show transfer improves with more alignment anchors.
- RISE/MDV-style prototype comparisons and spherical delta steering tests separate target-cosine accuracy from transformation-neighborhood retrieval.

Current status:

This is a stress-test and comparison track, not the main novelty claim. It is positioned relative to RISE, which is the closest neighboring work on geometric rotations for semantic-syntactic transformations.

Important result folder:

- `results/experiments/upat_large_results/`

### GLT-STEER: Activation Steering

Question:

> Can transformation vectors do behavior-level work inside a generative model?

Main evidence:

- A broad GPT-2/DistilGPT-2 pilot found that question steering was the clearest target for a focused rerun.
- The focused GPT-2 question-steering run completed `6800` generations with no failures.
- At layer `2`, gain `0.75`, target question steering produced question marks in `93.75%` of generations.
- Across all tested layers at gain `0.75`, target question steering produced question marks in `93.50%` of generations.
- Random-norm, wrong-class, and negative-target controls produced `0.00%` question marks in the aggregate control summary.
- A follow-up control found no-steering question-mark base rate `0.0000` for both in-template and out-of-template prompts.
- The same question vector transferred to 40 freeform out-of-template declarative sentences, producing question marks at rate `0.8375` while all compact controls stayed at `0.0000`.
- Prompt robustness controls show that the effect survives four prompt styles, with target question-mark rate `0.7750-0.9875` in-template and `0.8125-0.9250` out-of-template.
- Content-preservation audits show that prompt wording matters: copy-like prompts preserve source content while adding question form much better than bare or quoted prompts.
- Under copy-like prompts, target question steering reaches high question-and-preserved rates: `0.9625-0.9750` in-template and `0.8250-0.9000` out-of-template for `repeat_sentence`, `same_sentence`, and `copy_sentence`.
- A copy-prompt none-baseline audit shows that copy-like prompts without steering produce `0.0000` question marks across `960` no-steering rows from GPT-2 and DistilGPT-2.
- A DistilGPT-2 replication preserves the qualitative target-vs-control separation, but the effect is much weaker than GPT-2: best question-and-preserved rate `0.4625`.
- A first non-question extension to negation is negative: the current copy-prompt method does not show clean target-vs-control separation for negation.
- A harder out-of-template question audit with passive clauses, subordinate clauses, proper names, and numeric/time expressions preserves the question-form effect: target question-mark rate `0.7125-0.7500`, best matched control `0.0375`.
- A delta-coherence diagnostic explains part of the question/negation split: across all GPT-2 layers, question deltas are much more internally coherent than negation deltas. At the steering layers, mean pairwise cosine is `0.9693` vs `0.5621` on layer `2`, and `0.9365` vs `0.5665` on layer `3`.
- A full-layer negation sweep over GPT-2 layers `0-11` does not find a clean negation layer. The best target-and-preserved rate is `0.1729`, while controls also produce nontrivial rates up to `0.1229`.
- An exclamation-marker control supports the surface-marker explanation: `statement -> statement!` reaches exclamation-and-preserved rate `1.0000` in-template and `0.8000` on hard out-of-template sources, while no-steering remains `0.0000`.
- A GPT-2 vs DistilGPT-2 question-delta norm diagnostic shows that DistilGPT-2 question directions are not uniformly smaller, but they are strongly compressed in later relative layers: final relative-layer mean-norm ratio `0.3341`, centroid-norm ratio `0.2923`.

Current interpretation:

This is the first behavior-level intervention result in GLT. It shows that a question-transformation activation vector can steer GPT-2 toward question-like output form under residual-stream injection. The best prompt families preserve much of the source content while adding question form, but this is still not a complete semantic-editing system and not proof of a complete linguistic algebra. The failure analyses are also informative: question deltas are far more coherent than negation deltas in GPT-2 hidden space, final punctuation markers are much easier to steer than sentence-internal negation, and DistilGPT-2 appears to compress later-layer question directions.

Important result folders:

- `results/experiments/gpt2_activation_steering_pilot_results/`
- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`
- `results/experiments/gpt2_question_steering_controls_20260714_results/`
- `results/experiments/gpt2_question_prompt_robustness_20260715_results/`
- `results/experiments/gpt2_question_content_preservation_20260716_results/`
- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/`
- `results/experiments/distilgpt2_question_copy_prompt_preservation_20260716_results/`
- `results/experiments/question_copy_prompt_none_baseline_20260716_results/`
- `results/experiments/gpt2_negation_copy_prompt_steering_20260716_results/`
- `results/experiments/gpt2_question_hard_oot_copy_prompt_steering_20260716_results/`
- `results/experiments/gpt2_steering_delta_coherence_20260716_results/`
- `results/experiments/gpt2_negation_copy_prompt_layer_sweep_20260716_results/`
- `results/experiments/gpt2_exclamation_copy_prompt_steering_20260716_results/`
- `results/experiments/gpt2_distilgpt2_question_delta_norms_20260716_results/`

### GLT-AFFECT: Graded Affective Geometry

Question:

> Do graded emotional polarity scales form simple straight axes, or curved semantic geometry?

Current text-only scale:

```text
hate -> dislike -> indifferent -> like -> love
```

Main evidence:

- Adjacent affective steps are not uniform.
- `neutral -> love` and `neutral -> hate` are not opposite directions in text embeddings.
- Marker-only pooling and lexical-specificity controls weaken but preserve a small affect-leading signal.
- Bootstrap contrasts support a stable affect-specific excess over several lexical controls, but a substantial generic lexical-substitution component remains.

Current interpretation:

GLT-AFFECT is evidence about language-representation geometry, not grounded affective experience.

## Related Work Positioning

Closest neighboring work:

- Freenor and Alvarez 2026, RISE: geometric rotations for discourse-level semantic-syntactic transformations across languages and embedding models.
- Xia and Kalita 2025, Linear Relational Decoding of Morphology in Language Models: relation-specific matrix operators motivate GLT-MOLT.
- Park, Choe, and Veitch 2023/2024: Linear Representation Hypothesis framing for representation geometry.
- De Raedt et al. 2021: geometric cross-lingual linguistic transformations with pretrained autoencoders.

This repository's distinct angle is not "geometric transformations exist" in the broadest sense. Its contribution is a set of endpoint-controlled diagnostics, null baselines, signed-composition tests, operator-closure audits, and explicit negative/bounded results.

## Repository Map

- `paper/`
  - `research_program.md`: current research roadmap.
  - `research_roadmap.md`: research backlog and completed changes.
  - `related_work_positioning.md`: positioning against RISE, LRH, and related work.
  - `articles/`: draft paper candidates.
  - `figures/`: curated figures used in drafts and reports.
- `research/`
  - `diary.md`: dated research diary.
  - `PROTOCOL.md`: working protocol.
- `reports/`
  - PDF packets and release notes for external verification.
- `results/`
  - aggregate CSV summaries and experiment artifacts.
  - `results/experiments/`: experiment-specific CSVs, figures, metadata, and run summaries.
- `scripts/`
  - experiment scripts, report builders, figure builders, and post-hoc analysis scripts.

## Main Artifacts

- Research program: `paper/research_program.md`
- Track 1 draft: `paper/articles/geometric-transformation-vectors/draft.md`
- Track 2 / GLT-SPOT + GLT-MOLT draft: `paper/articles/lie-style-linguistic-operators/draft.md`
- Results index: `results/README.md`
- Main live report: `reports/2026-06-23_research_report.pdf`
- Zenodo snapshot report: `reports/2026-06-13_archival_report.pdf`

## Reproducibility

The datasets in this repository are synthetic controlled sentence-pair templates generated by scripts in `scripts/`; no external natural-language corpus is required for the archived experiments.

The current review environment is pinned in:

```text
requirements.txt
```

Run scripts from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe scripts\build_research_report.py
```

General reproducibility path:

1. Inspect or run scripts in `scripts/`.
2. Compare outputs against CSVs and figures in `results/` and `results/experiments/`.
3. Read the corresponding `RUN_SUMMARY.md` files in experiment folders.
4. Rebuild the research report if needed with `scripts/build_research_report.py`.

Large intermediate vector caches (`*.npy`, `*.npz`), local virtual environments, IDE files, `.env`, logs, and review zip archives are intentionally excluded from git.

## Citable Snapshots

The repository includes:

- `CITATION.cff` for GitHub's citation widget.
- `.zenodo.json` for Zenodo/GitHub release archiving.

Latest Zenodo version DOI:

- [10.5281/zenodo.20829303](https://doi.org/10.5281/zenodo.20829303)

Previous Zenodo version DOI:

- [10.5281/zenodo.20680414](https://doi.org/10.5281/zenodo.20680414)

Please cite this as a software/research-artifact snapshot, not as a peer-reviewed publication.

## License

MIT. See `LICENSE`.
