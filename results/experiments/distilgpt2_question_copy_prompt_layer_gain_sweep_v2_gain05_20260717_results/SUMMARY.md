# DistilGPT-2 question steering layer/gain sweep: gain 0.5

This run is the `gain=0.5` slice of a direct DistilGPT-2 layer/gain sweep for question steering.

## Setup

- Script: `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- Model: `distilgpt2`
- Transformation: `question`
- Layers: `0-5`
- Gain: `0.5`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Sources: `40` in-template test sources and `40` out-of-template freeform sources
- Raw generations: `7200`

## Main result

Layer-aware aggregation shows that `gain=0.5` is usable but not optimal.

Best target row:

| layer | source set | prompt style | question mark rate | question-and-preserved rate | rows |
|---:|---|---|---:|---:|---:|
| `1` | `out_of_template_freeform` | `same_sentence` | `0.6250` | `0.6250` | `40` |

Maximum target-and-preserved rates by control:

| control | max question-and-preserved rate |
|---|---:|
| `target` | `0.6250` |
| `random_norm` | `0.0500` |
| `negative_target` | `0.0250` |
| `none` | `0.0250` |
| `wrong_class` | `0.0250` |

## Interpretation

This already improves on the earlier aggregate DistilGPT-2 copy-prompt replication, but the best row is narrow: it depends on layer `1`, `same_sentence`, and the out-of-template source set.

## Files

- `csv/transformation_copy_prompt_summary.csv`: default aggregate summary from the runner.
- `csv/transformation_copy_prompt_raw.csv`: all generated outputs, including layer-level rows.
- `csv/transformation_copy_prompt_sources.csv`: source sentences.
- `run_status.json`: run metadata.
