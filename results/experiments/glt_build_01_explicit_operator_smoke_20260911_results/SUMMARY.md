# GLT-BUILD-01A Explicit-Operator Smoke Test

Status: completed smoke test. This is an implementation and pipeline validation run, not a promoted result.

## Audit Note Added 2026-09-15

The tables below preserve the pilot outputs. Evaluation used the first 96 examples
in each family, with uneven subject coverage. The three seeds reuse the same
sources; `n` is a generation count, not an independent-source count. Final reverse
pair accuracy is 37/288, all from QN; the other reversed pairs score zero.

Linear-map raw cosine was already approximately 0.9999 before training, while no
behavioral task was solved. High final cosine therefore does not demonstrate
learned transformation structure. The reverse-order endpoint cosine of 1 is
guaranteed by identical reference sentences in this commuting generator, not by
model execution. The script saved measurements at checkpoints but did not save
weights or hidden arrays. Its final-layer, gain-1 mean-vector interventions score
0/360 for target-vector conditions, which only bounds that tested recipe.

The corrected follow-up is specified in
[composition_audit_protocol.md](../../../paper/articles/glt-build-emergence/composition_audit_protocol.md).

## Configuration

- `started_at`: `2026-09-11 11:20:52 +0300`
- `out_dir`: `results\experiments\glt_build_01_explicit_operator_smoke_20260911_results`
- `seeds`: `0,1,2`
- `steps`: `800`
- `checkpoints`: `0,100,250,500,800`
- `batch_size`: `64`
- `eval_limit_per_group`: `96`
- `causal_limit_per_operation`: `24`
- `d_model`: `64`
- `layers`: `2`
- `heads`: `4`
- `dropout`: `0.0`
- `learning_rate`: `0.003`
- `ridge_alpha`: `1.0`
- `vocab_size`: `50`
- `train_examples`: `3600`
- `device`: `cpu`
- `torch_version`: `2.12.0+cpu`

## Final Behavioral Exact Match

| family | exact_match | n |
| --- | --- | --- |
| heldout_scene_reverse_seen_pair | 0.1285 | 288 |
| heldout_scene_seen_pair | 1.0000 | 288 |
| heldout_scene_single | 1.0000 | 288 |
| heldout_unseen_pair | 0.0000 | 288 |
| train_seen | 1.0000 | 288 |

## Final Representation Diagnostics

| layer | nearest_centroid_acc | source_target_cosine | additive_target_cosine | linear_target_cosine | affine_target_cosine | wrong_additive_target_cosine | random_additive_target_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.7644 | 0.7264 | 0.9876 | 0.9983 | 0.9983 | 0.7117 | 0.6348 |
| 2.0000 | 0.6674 | 0.5532 | 0.6757 | 0.9916 | 0.9917 | 0.5672 | 0.5417 |

## Final Composition Diagnostics

| pair | pair_seen_in_training | pair_vs_sum_cosine | sum_residual_ratio | reverse_order_delta_cosine | n |
| --- | --- | --- | --- | --- | --- |
| NQ | 1 | 0.7235 | 1.1716 | 1.0000 | 6 |
| NV | 0 | 0.4550 | 1.4430 | 1.0000 | 6 |
| SQ | 0 | 0.9979 | 0.0498 | 1.0000 | 6 |
| TN | 1 | 0.9083 | 0.3263 | 1.0000 | 6 |
| TQ | 1 | 0.9052 | 0.3452 | 1.0000 | 6 |
| TV | 0 | 0.8800 | 0.4497 | 1.0000 | 6 |
| VQ | 1 | 0.7373 | 1.2638 | 1.0000 | 6 |

## Final Causal Intervention Audit

| operation | condition | exact_match | identity_match | n |
| --- | --- | --- | --- | --- |
| N | negative_vector | 0.0000 | 1.0000 | 72 |
| N | none | 0.0000 | 1.0000 | 72 |
| N | random_norm | 0.0000 | 1.0000 | 72 |
| N | target_vector | 0.0000 | 0.9444 | 72 |
| N | wrong_vector | 0.0000 | 0.8611 | 72 |
| Q | negative_vector | 0.0000 | 1.0000 | 72 |
| Q | none | 0.0000 | 1.0000 | 72 |
| Q | random_norm | 0.0000 | 1.0000 | 72 |
| Q | target_vector | 0.0000 | 1.0000 | 72 |
| Q | wrong_vector | 0.0000 | 1.0000 | 72 |
| S | negative_vector | 0.0000 | 1.0000 | 72 |
| S | none | 0.0000 | 1.0000 | 72 |
| S | random_norm | 0.0000 | 1.0000 | 72 |
| S | target_vector | 0.0000 | 1.0000 | 72 |
| S | wrong_vector | 0.0000 | 1.0000 | 72 |
| T | negative_vector | 0.0000 | 1.0000 | 72 |
| T | none | 0.0000 | 1.0000 | 72 |
| T | random_norm | 0.0000 | 1.0000 | 72 |
| T | target_vector | 0.0000 | 1.0000 | 72 |
| T | wrong_vector | 0.0000 | 0.9444 | 72 |
| V | negative_vector | 0.0000 | 1.0000 | 72 |
| V | none | 0.0000 | 1.0000 | 72 |
| V | random_norm | 0.0000 | 1.0000 | 72 |
| V | target_vector | 0.0000 | 0.8611 | 72 |
| V | wrong_vector | 0.0000 | 1.0000 | 72 |

## Interpretation Guardrail

This smoke test validates the GLT-BUILD pipeline shape: synthetic data generation, tiny-model training, checkpointed behavioral evaluation, representation diagnostics, composition diagnostics, and a first causal intervention audit.

The run should not be cited as evidence that a stable transformation algebra has emerged. A promoted GLT-BUILD result requires multiple seeds, a larger fixed protocol, and explicit null comparisons.
