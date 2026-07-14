# GPT-2 Question Activation Steering Summary

## Status

Completed successfully.

- Started: `2026-07-14 01:18:38 +0300`
- Finished: `2026-07-14 03:04:04 +0300`
- Model: `gpt2`
- Evaluation class: `question`
- Residual layers: `2, 3, 4, 5, 6`
- Gains: `0.25, 0.5, 0.75, 1.0`
- Controls: `none`, `target`, `wrong_class`, `random_norm`, `negative_target`
- Raw rows: `6800`
- Summary rows: `85`
- Failures: none

## Question

Can a transformation vector derived from representation deltas do behavior-level work when injected back into a generative transformer?

This focused run tests the question-formation direction. It computes transformation centroids from hidden-state deltas and adds a selected vector into GPT-2 residual activations during generation.

## Main Result

The question target vector strongly steers GPT-2 toward question-like generation, while random, wrong-class, and negative controls do not produce question marks in the aggregate control summary.

Overall across layers at gain `0.75`:

| control | target marker rate | question mark rate |
|---|---:|---:|
| target | `0.9550` | `0.9350` |
| random_norm | `0.1350` | `0.0000` |
| wrong_class | `0.1050` | `0.0000` |
| negative_target | `0.1900` | `0.0000` |

Best individual layer/gain setting:

| layer | gain | target marker rate | question mark rate | best control marker rate | margin |
|---:|---:|---:|---:|---:|---:|
| `2` | `0.75` | `1.0000` | `0.9375` | `0.1125` | `0.8875` |

Question-mark rate for target steering:

| layer | gain 0.50 | gain 0.75 | gain 1.00 |
|---:|---:|---:|---:|
| `2` | `0.3875` | `0.9375` | `0.6500` |
| `3` | `0.3000` | `0.9875` | `0.9000` |
| `4` | `0.3250` | `0.9375` | `0.6500` |
| `5` | `0.1125` | `0.9375` | `0.9250` |
| `6` | `0.0250` | `0.8750` | `0.8000` |

## Example Outputs

Layer `2`, gain `0.75`.

Source:

```text
The robot opened the portal.
```

No steering:

```text
The robot opened the portal. The robot opened the portal. The portal opened...
```

Target question steering:

```text
The robot opened the portal. The robot opened the portal? The robot opened the portal?
```

Random-norm control:

```text
The robot opened the portal. The robot opened the portal. The robot opened the portal...
```

Wrong-class control:

```text
The robot opened the portal. The robot opened the portal. The portal opened...
```

Negative-target control:

```text
The robot was able to enter the portal. The robot was able to enter the portal...
```

## Interpretation

This is the first behavior-level intervention result in GLT: a transformation vector derived from representation deltas can be injected back into GPT-2 and measurably steer generation toward the corresponding linguistic form.

The strongest safe claim is:

> A question-transformation activation vector can steer GPT-2 toward question-mark generation under controlled residual-stream injection, while random, wrong-class, and negative controls do not show the same behavior.

## Caveats

- This is a focused GPT-2-only result for question formation.
- The outputs are not clean sentence rewrites; they often repeat the prompt sentence.
- The question-mark metric shows strong form steering, not full semantic preservation.
- Other transformation classes were weak in the broader pilot and need redesigned prompts or metrics.
- This is intervention-style evidence, not proof of a complete linguistic algebra.

## Files

- Script: `scripts/run_gpt2_activation_steering_pilot.py`
- Launcher: `scripts/launch_gpt2_activation_steering_pilot.ps1`
- Status: `run_status.json`
- Dataset: `csv/activation_steering_dataset.csv`
- Raw generations: `csv/activation_steering_raw.csv`
- Summary table: `csv/activation_steering_summary.csv`

