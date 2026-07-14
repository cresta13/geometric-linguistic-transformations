# GPT-2 Activation Steering Pilot Summary

## Status

Completed successfully.

- Started: `2026-07-11 00:55:15 +0300`
- Finished: `2026-07-11 02:45:40 +0300`
- Models: `distilgpt2`, `gpt2`
- Classes: `question`, `negation`, `modality`, `tense_shift`
- Residual layers:
  - `distilgpt2`: lower/middle/final available layers
  - `gpt2`: `4`, `8`, `11`
- Gains: `0.75`, `1.5`, `3.0`
- Controls: `none`, `target`, `wrong_class`, `random_norm`, `negative_target`
- Raw rows: `7488`
- Summary rows: `312`
- Failures: none

## Purpose

This was the first broad generation-time activation-steering pilot for GLT. It asked whether transformation delta centroids can be injected into GPT-2-family residual streams to make generated text acquire the target transformation.

## Main Finding

The broad pilot was mostly noisy, but it identified question formation in GPT-2 as the clearest promising case.

Best broad-pilot question row:

| model | class | control | gain | target marker rate |
|---|---|---|---:|---:|
| `gpt2` | `question` | `target` | `0.75` | `0.3333` |

Other classes were weak under this first prompt and metric setup. This is why the next run narrowed to a focused GPT-2 question-steering experiment with smaller gains, more test sources, and an explicit question-mark metric.

## Interpretation

The pilot should be treated as exploratory. Its value is not a standalone claim; its value is that it selected the first transformation and gain range for a cleaner follow-up.

The follow-up result is:

- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`

## Caveats

- The broad marker metric can count some loose lexical markers that are not clean transformation success.
- Negation, modality, and tense shift did not show strong steering under this setup.
- The run did not include the explicit `question_mark_hit` metric added for the focused question rerun.
- This pilot should not be cited without the focused follow-up.

## Files

- Script: `scripts/run_gpt2_activation_steering_pilot.py`
- Launcher: `scripts/launch_gpt2_activation_steering_pilot.ps1`
- Status: `run_status.json`
- Dataset: `csv/activation_steering_dataset.csv`
- Raw generations: `csv/activation_steering_raw.csv`
- Summary table: `csv/activation_steering_summary.csv`

