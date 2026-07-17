# DistilGPT-2 question steering layer/gain sweep: gain 1.5

This run is the `gain=1.5` slice of a direct DistilGPT-2 layer/gain sweep for question steering.

## Setup

- Script: `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- Model: `distilgpt2`
- Transformation: `question`
- Layers: `0-5`
- Gain: `1.5`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Sources: `40` in-template test sources and `40` out-of-template freeform sources
- Raw generations: `7200`

## Main result

The stronger gain over-steers and damages content preservation.

Best target row:

| layer | source set | prompt style | question mark rate | question-and-preserved rate | rows |
|---:|---|---|---:|---:|---:|
| `2` | `in_template_test` | `repeat_sentence` | `0.8500` | `0.0750` | `40` |

Maximum target-and-preserved rates by control:

| control | max question-and-preserved rate |
|---|---:|
| `negative_target` | `0.2250` |
| `target` | `0.0750` |
| `random_norm` | `0.0500` |
| `none` | `0.0250` |
| `wrong_class` | `0.0250` |

## Interpretation

`gain=1.5` is too strong for this setup. It can still induce question markers, but content preservation collapses, and the negative-target control can outperform the target on the joint metric. This supports treating gain as an intervention parameter that must be tuned, not as a harmless scale factor.

## Files

- `csv/transformation_copy_prompt_summary.csv`: default aggregate summary from the runner.
- `csv/transformation_copy_prompt_raw.csv`: all generated outputs, including layer-level rows.
- `csv/transformation_copy_prompt_sources.csv`: source sentences.
- `run_status.json`: run metadata.
