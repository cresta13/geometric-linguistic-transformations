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
  - `csv/procrustes_null_summary.csv`: pilot null baselines for cross-model Procrustes transfer.
  - `figures/10_procrustes_null_random_pairing.png` and `figures/10_procrustes_null_random_labels.png`: observed aligned F1 versus null baselines.

The paper drafts link directly to the relevant CSVs and figures in these folders.
