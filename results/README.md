# Results Index

This directory contains compact CSV summaries and experiment artifacts.

## Aggregate Tables

- `ablation_control_table.csv`: main multiseed delta/y_only/concat summary.
- `ablation_multiseed_aggregated.csv`: aggregated multiseed ablation table.
- `ablation_multiseed_mcnemar.csv`: McNemar tests for delta vs endpoint baselines.
- `track1_multiseed_effect_intervals.csv`: 95% seed-level intervals for `delta-y_only` and `delta-concat`.
- `confusion_negation_summary.csv`: negation vs non-negation confusion summary.
- `confusion_class_recall.csv`: per-class recall from full-semantic confusion matrices.

## Experiment Artifacts

Detailed experiment outputs live under `results/experiments/`.

Important subdirectories:

- `lie_llm_full_semantic_holdout_results/`
- `lie_llm_syntax_results/`
- `syntax_representation_ablation_results/`
- `full_semantic_pooling_ablation_results/`
- `track1_spotcheck_results/`
- `track1_spotcheck_large_results/`
- `lie_composition_results/`
- `lie_composition_grammar_results/`
  - `csv/grammar_composition_dataset.csv`: grammar-generated pairwise composition dataset for `N,Q,M,T`.
  - `csv/grammar_composition_summary.csv`: relative commutator norm and cosine summaries.
  - `csv/grammar_endpoint_controls.csv`: endpoint-only and delta-only pair-label controls.
  - `csv/grammar_commutator_nulls.csv`: `N=1000` same-pair, any-pair, and norm-matched random-direction commutator nulls.
  - `figures/01_relative_commutator_norm_heatmap.png`: grammar-control relative commutator heatmap.
- `lie_multilingual_max_results/`
  - `csv/multilingual_composition_dataset.csv`: 7-language synthetic composition dataset for `N,Q,M,T`.
  - `csv/triple_global_summary.csv`, `csv/triple_by_model_summary.csv`, and `csv/triple_by_language_summary.csv`: third-order signed-permutation summaries across multilingual encoders.
  - `csv/pair_global_summary.csv`: pairwise commutator summary across models and languages.
  - `csv/endpoint_controls_summary.csv`: held-out-language source/endpoint/delta/commutator controls.
  - `csv/cross_language_centroid_summary.csv`: cross-language centroid-consistency summary.
  - `figures/01_multilingual_signed_permutation_ratios.png`, `figures/02_triple_global_ratio_summary.png`, `figures/03_endpoint_control_macro_f1.png`, and `figures/04_cross_language_centroid_consistency.png`: compact figures for the multilingual audit.
- `lie_endpoint_subspace_9m_96t_pca128_results/`
  - `RUN_SUMMARY.md`: 2026-06-29 9-model endpoint-subspace residualization summary.
  - `csv/subspace_residualized_global_summary.csv`: global raw and residualized signed-null ratios across 9 models, 7 languages, and 4 triples.
  - `csv/endpoint_subspace_probe_summary.csv`: endpoint-derived probe checks for triple label, endpoint position, and cyclic sign.
  - `figures/01_subspace_residualization_ratios.png`: compact residualization-ratio figure.
- `glt_molt_affine_operator_9m_160t_300null_results/`
  - `RUN_SUMMARY.md`: 2026-06-29 9-model GLT-MOLT affine/operator summary.
  - `csv/operator_fit_summary.csv`: additive, linear, and affine target-reconstruction results.
  - `csv/composition_prediction_summary.csv`: pairwise composition-prediction summary.
  - `csv/matrix_closure_summary.csv`: matrix-commutator closure against random-subspace nulls.
  - `csv/matrix_jacobi_summary.csv`: Jacobi-like operator-norm diagnostics.
  - `figures/matrix_closure_linear.png` and `figures/matrix_closure_affine.png`: closure residual figures.
- `glt_molt_affine_operator_9m_160t_1000null_results/`
  - `RUN_SUMMARY.md`: 2026-06-30 9-model GLT-MOLT affine/operator confirmation with 1000 random-subspace nulls.
  - `csv/operator_fit_summary.csv`: additive, linear, and affine target-reconstruction results.
  - `csv/composition_prediction_summary.csv`: pairwise composition-prediction summary.
  - `csv/matrix_closure_summary.csv`: matrix-commutator closure against random-subspace nulls.
  - `csv/matrix_jacobi_summary.csv`: Jacobi-like operator-norm diagnostics.
  - `figures/matrix_closure_linear.png` and `figures/matrix_closure_affine.png`: closure residual figures.
- `glt_molt_ridge_sweep_9m_160t_300null_results/`
  - `RUN_SUMMARY.md`: 2026-07-01 GLT-MOLT ridge-alpha sweep summary.
  - `csv/operator_fit_by_alpha_method.csv`: one-step target prediction by ridge alpha and method.
  - `csv/matrix_closure_by_alpha_method.csv`: matrix-commutator closure by ridge alpha and method.
  - `csv/matrix_jacobi_by_alpha_method.csv`: Jacobi-like operator norm by ridge alpha and method.
- `glt_molt_matched_nulls_9m_160t_a10_100_1000null_results/`
  - `RUN_SUMMARY.md`: 2026-07-02 GLT-MOLT matched-null operator-closure summary.
  - `csv/matched_closure_by_alpha_method.csv`: closure residuals versus random-subspace, Gaussian norm-matched, and signed-permutation matched nulls.
  - `csv/operator_fit_by_alpha_method.csv`: one-step target prediction for `alpha=10` and `alpha=100`.
  - `csv/operator_norms_summary.csv`: learned operator norm summaries by alpha, method, and primitive operator.
- `glt_molt_spectral_nulls_9m_160t_a100_300null_g256_results/`
  - `RUN_SUMMARY.md`: 2026-07-03 GLT-MOLT spectral-null operator-closure summary.
  - `csv/spectral_closure_by_alpha_method.csv`: `alpha=100` closure residuals versus singular-spectrum matched Givens-rotation nulls.
  - `csv/spectral_closure_summary.csv`: pairwise commutator closure summaries by method and ordered-pair difference.
  - `csv/operator_fit_by_alpha_method.csv`: one-step target prediction for the same learned linear and affine operators.
- `glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256_results/`
  - `RUN_SUMMARY.md`: 2026-07-08 compact GLT-MOLT PCA-64 spectral-null sensitivity summary.
  - `pca_64/csv/spectral_closure_by_alpha_method.csv`: completed five-model PCA-64 closure residuals versus singular-spectrum matched nulls.
  - `pca_64/csv/operator_fit_by_alpha_method.csv`: PCA-64 one-step target prediction for the same learned operators.
- `glt_molt_spectral_pca128_5m_160t_a100_300null_g256_results/`
  - `RUN_SUMMARY.md`: 2026-07-08 compact GLT-MOLT PCA-128 spectral-null sensitivity summary.
  - `csv/spectral_closure_by_alpha_method.csv`: completed five-model PCA-128 closure residuals versus singular-spectrum matched nulls.
  - `csv/operator_fit_by_alpha_method.csv`: PCA-128 one-step target prediction for the same learned operators.
- `lie_composition_decoder_results/`
- `lie_algebraic_identities_results/`
- `lie_algebraic_identities_decoder_results/`
- `lie_semantic_equivalence_results/`
- `upat_audit_results/`
- `upat_large_results/`
  - `csv/procrustes_null_raw.csv`: `N=1000` raw null repeats for cross-model Procrustes transfer (`30` directions x `3` nulls x `1000` repeats).
  - `csv/procrustes_null_summary.csv`: `N=1000` random-label, random-pairing, and random-orthogonal null summaries.
  - `figures/10_procrustes_null_random_pairing.png`, `figures/10_procrustes_null_random_labels.png`, and `figures/10_procrustes_null_random_orthogonal.png`: observed aligned F1 versus null baselines.
  - `csv/heldout_alignment_anchor_texts.csv`: auxiliary anchor texts disjoint from the classifier train/test texts.
  - `csv/heldout_alignment_curve_raw.csv`, `csv/heldout_alignment_curve_summary.csv`, and `csv/heldout_alignment_curve_by_direction.csv`: held-out anchor alignment-size curve across all non-identity cross-model directions.
  - `figures/11_heldout_alignment_size_curve.png` and `figures/11_heldout_alignment_by_direction.png`: held-out alignment-size and by-direction summaries.
  - `csv/rise_aware_comparison_raw.csv` and `csv/rise_aware_comparison_summary.csv`: first-pass MDV/RISE-style prototype comparison on UPAT.
  - `figures/12_rise_aware_target_cosine.png` and `figures/12_rise_aware_retrieval_f1.png`: target-prediction and nearest-target class-retrieval summaries.
  - `csv/hybrid_rise_procrustes_raw.csv` and `csv/hybrid_rise_procrustes_summary.csv`: non-leaky hybrid RISE-Procrustes transfer test using all-class prototype score features.
  - `figures/13_hybrid_rise_procrustes_f1.png` and `figures/13_hybrid_rise_procrustes_heatmap.png`: hybrid transformation-label F1 summaries.
  - `csv/spherical_delta_steering_raw.csv` and `csv/spherical_delta_steering_summary.csv`: movement-level linear/spherical/RISE steering comparison.
  - `figures/14_spherical_delta_target_cosine.png`, `figures/14_spherical_delta_retrieval_top1.png`, and `figures/14_spherical_delta_retrieval_label_f1.png`: target-cosine and retrieval summaries for spherical delta steering.
- `gpt2_activation_steering_pilot_results/`
  - `SUMMARY.md`: exploratory pilot summary and rationale for the focused question rerun.
  - exploratory GPT-2/DistilGPT-2 generation-time residual-stream steering pilot over question, negation, modality, and tense-shift vectors.
  - `csv/activation_steering_raw.csv`: generated outputs and marker hits for target, wrong-class, random-norm, and negative-vector controls.
  - `csv/activation_steering_summary.csv`: aggregate marker rates by model, layer, class, control, and gain.
- `gpt2_question_activation_steering_focused_20260714_results/`
  - `SUMMARY.md`: focused GPT-2 question-steering result summary.
  - `csv/activation_steering_raw.csv`: 6800 generated outputs for GPT-2 question steering.
  - `csv/activation_steering_summary.csv`: 85-row summary table.
  - `run_status.json`: completed run metadata.
- `gpt2_question_steering_controls_20260714_results/`
  - `SUMMARY.md`: base-rate and out-of-template question-steering control summary.
  - `csv/question_mark_base_rate.csv`: no-steering question-mark base rates for in-template and freeform prompts.
  - `csv/question_steering_controls_summary.csv`: compact target/control comparison for in-template and out-of-template prompts.
  - `csv/question_steering_controls_raw.csv`: 1200 generated outputs across layers `2,3`, gain `0.75`, and controls.
  - `csv/question_steering_control_sources.csv`: in-template and freeform source prompts.
- `gpt2_question_prompt_robustness_20260715_results/`
  - `SUMMARY.md`: prompt-wrapper robustness summary for GPT-2 question steering.
  - `csv/question_prompt_robustness_summary.csv`: question-mark rates by source set, prompt style, and control.
  - `csv/question_prompt_robustness_raw.csv`: 3200 generated outputs across prompt styles, layers `2,3`, and controls.
  - `csv/question_prompt_robustness_sources.csv`: in-template and freeform source prompts.
- `gpt2_question_content_preservation_20260716_results/`
  - `SUMMARY.md`: first content-preservation audit for the GPT-2 question-steering result.
  - `csv/question_content_preservation_summary.csv`: question-mark and source-content preservation rates by prompt style and control.
  - `csv/question_content_preservation_contrast.csv`: target-vs-best-control contrast summary.
  - `csv/question_content_preservation_raw.csv`: generated outputs with content-preservation flags.
- `gpt2_question_copy_prompt_preservation_20260716_results/`
  - `SUMMARY.md`: copy-prompt content-preservation follow-up for GPT-2 question steering.
  - `csv/question_copy_prompt_summary.csv`: question-mark, content-preservation, and joint rates by prompt style and control.
  - `csv/question_copy_prompt_sources.csv`: in-template and freeform source prompts.
  - `csv/question_copy_prompt_raw.csv`: 3200 generated outputs across copy-oriented prompt styles, layers `2,3`, and controls.
  - `run_status.json`: completed run metadata.
- `distilgpt2_question_copy_prompt_preservation_20260716_results/`
  - `SUMMARY.md`: DistilGPT-2 replication of the copy-prompt question-steering preservation audit.
  - `csv/question_copy_prompt_summary.csv`: question-mark, content-preservation, and joint rates by prompt style and control.
  - `csv/question_copy_prompt_sources.csv`: in-template and freeform source prompts.
  - `csv/question_copy_prompt_raw.csv`: 3200 generated outputs across copy-oriented prompt styles, layers `2,3`, and controls.
  - `run_status.json`: completed run metadata.
- `question_copy_prompt_none_baseline_20260716_results/`
  - `SUMMARY.md`: explicit no-steering copy-prompt baseline audit for GPT-2 and DistilGPT-2.
  - `csv/copy_prompt_none_baseline_rows.csv`: all no-steering copy-like prompt rows from the two copy-prompt runs.
  - `csv/copy_prompt_none_baseline_summary.csv`: max/mean no-steering question-mark rates by model and source set.
  - `csv/copy_prompt_none_baseline_global.csv`: global prompt-only control summary.
- `gpt2_negation_copy_prompt_steering_20260716_results/`
  - `SUMMARY.md`: first non-question GLT-STEER extension attempt, showing weak/negative negation steering under copy-like prompts.
  - `csv/transformation_copy_prompt_summary.csv`: negation target/control marker and preservation rates by source set and prompt style.
  - `csv/transformation_copy_prompt_raw.csv`: 2400 generated outputs across layers `2,3`, copy-like prompt styles, and controls.
  - `csv/transformation_copy_prompt_sources.csv`: in-template and freeform source prompts.
  - `run_status.json`: completed run metadata.

The paper drafts link directly to the relevant CSVs and figures in these folders.
