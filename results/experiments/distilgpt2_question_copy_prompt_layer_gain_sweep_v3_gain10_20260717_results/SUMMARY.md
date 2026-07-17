# DistilGPT-2 question steering layer/gain sweep: gain 1.0

This run is the `gain=1.0` slice of a direct DistilGPT-2 layer/gain sweep for question steering.

## Setup

- Script: `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- Model: `distilgpt2`
- Transformation: `question`
- Layers: `0-5`
- Gain: `1.0`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Sources: `40` in-template test sources and `40` out-of-template freeform sources
- Raw generations: `7200`

## Main result

Layer-aware aggregation shows that DistilGPT-2 can produce a strong question-steering effect when layer and gain are selected directly.

Best target rows:

| layer | source set | prompt style | question mark rate | question-and-preserved rate | rows |
|---:|---|---|---:|---:|---:|
| `2` | `in_template_test` | `same_sentence` | `1.0000` | `0.8250` | `40` |
| `1` | `in_template_test` | `same_sentence` | `0.7750` | `0.7500` | `40` |
| `2` | `in_template_test` | `repeat_sentence` | `0.8750` | `0.6750` | `40` |
| `1` | `out_of_template_freeform` | `same_sentence` | `0.8000` | `0.6250` | `40` |

Maximum target-and-preserved rates by control:

| control | max question-and-preserved rate |
|---|---:|
| `target` | `0.8250` |
| `negative_target` | `0.0500` |
| `random_norm` | `0.0500` |
| `none` | `0.0250` |
| `wrong_class` | `0.0250` |

## Interpretation

This changes the DistilGPT-2 story. The earlier weak aggregate replication was not simply a hard model failure. DistilGPT-2 is strongly layer/gain sensitive: at `gain=1.0`, layer `2`, and copy-like prompts, the effect becomes large and clean against controls.

This does not erase the architecture-dependence caveat. The effect is narrower than GPT-2 and degrades under over-strong gain, but the direct sweep shows that DistilGPT-2 can support strong question steering at the right intervention site.

## Files

- `csv/transformation_copy_prompt_summary.csv`: default aggregate summary from the runner.
- `csv/transformation_copy_prompt_raw.csv`: all generated outputs, including layer-level rows.
- `csv/distilgpt2_gain_layer_sweep_aggregate.csv`: layer-aware aggregate across gains `0.5`, `1.0`, and `1.5`.
- `csv/transformation_copy_prompt_sources.csv`: source sentences.
- `run_status.json`: run metadata.
