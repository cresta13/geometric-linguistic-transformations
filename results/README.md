# Results Index

This directory contains compact CSV summaries and experiment artifacts.

## Aggregate Tables

- `reviewer_ablation_table.csv`: main multiseed delta/y_only/concat summary.
- `ablation_multiseed_aggregated.csv`: aggregated multiseed ablation table.
- `ablation_multiseed_mcnemar.csv`: McNemar tests for delta vs endpoint baselines.
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

The paper drafts link directly to the relevant CSVs and figures in these folders.
