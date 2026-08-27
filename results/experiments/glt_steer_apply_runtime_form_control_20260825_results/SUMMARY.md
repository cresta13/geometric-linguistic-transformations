# GLT-STEER Runtime Form-Control Applicability Audit

Status: completed successfully.

Run completed: 2026-08-25  
Reviewed: 2026-08-27

## Purpose

This audit tests whether GLT-STEER has a plausible practical use as runtime output-form control, not only as a mechanistic final-marker probe.

The key question is:

> If the desired transformation is a final-position marker (`?`, `!`, or `...`), does activation steering give useful control compared with prompt-only and deterministic baselines?

This is an applicability audit, not a new layer/gain search.

## Configuration

- Script: `scripts/run_glt_steer_apply_runtime_form_control.py`
- Result directory: `results/experiments/glt_steer_apply_runtime_form_control_20260825_results/`
- Models: `gpt2`, `distilgpt2`
- Targets: `question`, `exclamation`, `ellipsis`
- Held-out sources: `80`
- Training pairs: `360`
- Prompt styles: `neutral_restate`, `neutral_same`
- Controls: `none`, `strong_prompt`, `string_append_source`, `target`, `wrong_marker`, `random_norm`, `negative_target`
- Raw rows: `8640`
- Failures: none

Model settings were fixed before this run:

- GPT-2: layers `2,3`, gain `0.75`
- DistilGPT-2: layer `2`, gain `1.0`

## Headline Global Results

The table reports aggregate target-steering rows. Wilson 95% confidence intervals are shown for marker and marker-plus-content rates.

| model | target | N | target marker rate | target marker+content | malformed/repetitive |
|---|---|---:|---:|---:|---:|
| `gpt2` | `question` | `320` | `0.6281` CI `[0.574, 0.679]` | `0.3469` CI `[0.297, 0.401]` | `0.3469` |
| `gpt2` | `exclamation` | `320` | `0.6031` CI `[0.549, 0.655]` | `0.3844` CI `[0.333, 0.439]` | `0.2344` |
| `gpt2` | `ellipsis` | `320` | `0.6937` CI `[0.641, 0.742]` | `0.3219` CI `[0.273, 0.375]` | `0.3000` |
| `distilgpt2` | `question` | `160` | `0.4938` CI `[0.417, 0.570]` | `0.1688` CI `[0.119, 0.234]` | `0.6125` |
| `distilgpt2` | `exclamation` | `160` | `0.8438` CI `[0.780, 0.892]` | `0.4250` CI `[0.351, 0.502]` | `0.1938` |
| `distilgpt2` | `ellipsis` | `160` | `0.7250` CI `[0.651, 0.788]` | `0.2562` CI `[0.195, 0.329]` | `0.5687` |

Matched vector controls (`none`, `wrong_marker`, `random_norm`, and `negative_target`) produce `0.0000` marker-plus-content rate in every model/target aggregate cell. The `strong_prompt` baseline also produces `0.0000` target-marker rate in every model/target cell under this generation protocol.

The deterministic `string_append_source` baseline is trivially perfect: marker rate `1.0000`, content preservation `1.0000`, marker-plus-content `1.0000`, and malformed/repetitive rate `0.0000`.

## Best Target Rows

The best target rows by model/target are:

| model | target | prompt style | layer | marker | content | marker+content | malformed/repetitive | median marker rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | `question` | `neutral_restate` | `3` | `0.6750` | `0.6000` | `0.3750` | `0.1375` | `1.0` |
| `gpt2` | `exclamation` | `neutral_same` | `2` | `0.6875` | `0.5625` | `0.4250` | `0.3125` | `1.0` |
| `gpt2` | `ellipsis` | `neutral_same` | `3` | `0.7625` | `0.6125` | `0.4500` | `0.3250` | `1.0` |
| `distilgpt2` | `question` | `neutral_restate` | `2` | `0.6500` | `0.5625` | `0.2500` | `0.4625` | `1.0` |
| `distilgpt2` | `exclamation` | `neutral_restate` | `2` | `0.7875` | `0.6625` | `0.4625` | `0.1875` | `1.0` |
| `distilgpt2` | `ellipsis` | `neutral_same` | `2` | `0.7125` | `0.3750` | `0.2875` | `0.7250` | `1.0` |

## Interpretation

This is a positive activation-space control result but a bounded practical result.

Positive:

- Target steering produces the requested final marker where no-steering, strong-prompt, wrong-marker, random-norm, and negative-vector controls do not.
- The target marker usually reaches median rank `1.0` in the best target rows.
- The result transfers across GPT-2 and DistilGPT-2, though with model/marker-dependent quality.

Boundary:

- The deterministic `string_append_source` baseline dominates the practical final-marker task.
- Target steering often damages output quality, especially for DistilGPT-2 question and ellipsis.
- Therefore this audit does not justify claiming that GLT-STEER is a better production method for adding final punctuation.

Defensible claim:

> GLT-STEER is useful as an inference-time activation-space diagnostic and form-bias intervention for final-position markers. For known, deterministic final-marker edits, ordinary postprocessing remains simpler, stronger, and cleaner.

This strengthens the paper by clarifying applicability: the main value of GLT-STEER is mechanistic and diagnostic, with possible runtime-control relevance only when direct string postprocessing is unavailable, undesirable, or not equivalent to the intervention being studied.
