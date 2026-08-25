# GLT-STEER Fixed-Parameter Confirmatory Audit

Status: completed successfully on 2026-08-25.

This run tests whether the current GLT-STEER Final Marker Hypothesis survives a fixed-parameter confirmatory rerun on fresh hard-heldout sources. No layer or gain search is performed inside this run.

## Configuration

- Script: `scripts/run_glt_steer_confirmatory_fixed_params.py`
- Models: `gpt2`, `distilgpt2`
- Target classes: `question`, `exclamation`, `ellipsis`
- Source split: `48` fresh hard-heldout sources
- Training pairs: `360`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Controls: `none`, `target`, `wrong_marker`, `random_norm`, `negative_target`
- GPT-2 fixed setting: layers `2,3`, gain `0.75`
- DistilGPT-2 fixed setting: layer `2`, gain `1.0`
- Raw rows: `6480`
- Summary rows: `135`
- Failures: none

## Headline Aggregates

Rates below are aggregated from raw generation rows. Wilson 95% confidence intervals are shown for target rows.

| model | target | target marker rate | max control marker | target marker+preserved | max control marker+preserved |
|---|---|---:|---:|---:|---:|
| `gpt2` | `question` | `0.6562` (`N=288`, CI `[0.600, 0.709]`) | `0.0000` | `0.2396` (`N=288`, CI `[0.194, 0.292]`) | `0.0000` |
| `gpt2` | `exclamation` | `0.6875` (`N=288`, CI `[0.632, 0.738]`) | `0.0000` | `0.3958` (`N=288`, CI `[0.341, 0.453]`) | `0.0000` |
| `gpt2` | `ellipsis` | `0.8438` (`N=288`, CI `[0.797, 0.881]`) | `0.0000` | `0.3854` (`N=288`, CI `[0.331, 0.443]`) | `0.0000` |
| `distilgpt2` | `question` | `0.6319` (`N=144`, CI `[0.551, 0.706]`) | `0.0069` | `0.0556` (`N=144`, CI `[0.028, 0.106]`) | `0.0000` |
| `distilgpt2` | `exclamation` | `0.8958` (`N=144`, CI `[0.835, 0.936]`) | `0.0139` | `0.3472` (`N=144`, CI `[0.274, 0.428]`) | `0.0000` |
| `distilgpt2` | `ellipsis` | `0.8333` (`N=144`, CI `[0.764, 0.885]`) | `0.0000` | `0.1597` (`N=144`, CI `[0.109, 0.228]`) | `0.0000` |

## Interpretation

The fixed-parameter run supports the narrow Final Marker Hypothesis: final-position markers remain steerable on fresh hard-heldout sources, and matched no-steering, wrong-marker, random-norm, and negative-vector controls stay near zero.

This is strongest as a form-steering result. Strict marker-plus-content preservation is positive but modest, especially for DistilGPT-2 question steering. The run therefore strengthens the no-new-tuning evidence for final-marker steering, but it should not be reported as general semantic editing.

## Files

- `run_status.json`: run metadata
- `csv/glt_steer_confirmatory_raw.csv`: raw generations and marker/content flags
- `csv/glt_steer_confirmatory_summary.csv`: aggregate rates by model, target, prompt, layer, and control
- `csv/glt_steer_confirmatory_sources.csv`: heldout source sentences
- `csv/glt_steer_confirmatory_training_pairs.csv`: synthetic training pairs
- `csv/glt_steer_confirmatory_marker_token_ids.json`: marker token IDs used by model/target
