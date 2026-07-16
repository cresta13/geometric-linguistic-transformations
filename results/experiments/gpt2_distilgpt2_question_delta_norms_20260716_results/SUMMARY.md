# GPT-2 vs DistilGPT-2 question delta norm diagnostic

This diagnostic tests whether the weaker DistilGPT-2 steering result can be partly explained by compressed or flattened question-transformation directions.

## Setup

- Script: `scripts/analyze_gpt2_distilgpt2_question_delta_norms.py`
- Models: `gpt2`, `distilgpt2`
- Transformation: `question`
- Rows: `80` matched question pairs
- Layers: all available GPT-style transformer blocks for each model
- Metrics: delta norm, centroid norm, centroid-to-mean norm, and pairwise delta coherence

## Main result

DistilGPT-2 does not have uniformly smaller question deltas. Early layers are comparable or larger in DistilGPT-2, but mid-to-late layers are substantially smaller than GPT-2 at matched relative depth.

Relative-layer comparison:

| relative layer | GPT-2 layer | DistilGPT-2 layer | mean-norm ratio | centroid-norm ratio |
|---:|---:|---:|---:|---:|
| `0.0` | `0` | `0` | `1.1701` | `1.2056` |
| `0.2` | `2` | `1` | `1.0924` | `1.0015` |
| `0.4` | `4` | `2` | `0.8788` | `0.8675` |
| `0.6` | `7` | `3` | `0.6210` | `0.6150` |
| `0.8` | `9` | `4` | `0.6377` | `0.6328` |
| `1.0` | `11` | `5` | `0.3341` | `0.2923` |

## Interpretation

The result partially supports the compression explanation for the weaker DistilGPT-2 steering effect. The relevant question direction is not globally smaller in DistilGPT-2, but it is much smaller in later relative layers, where autoregressive behavior may be more directly exposed to generation.

This is not a complete explanation of the `0.975` vs `0.4625` joint-metric gap. The next stronger test would sweep DistilGPT-2 layers and gains directly, using the same copy-prompt preservation metric, rather than inferring from delta norms alone.

## Files

- `csv/question_delta_norm_summary.csv`: model/layer delta norm and coherence metrics.
- `csv/gpt2_distilgpt2_relative_layer_comparison.csv`: matched relative-depth comparison.
- `csv/question_delta_norm_raw.csv`: per-row delta norms.
- `csv/question_delta_sources.csv`: matched source/target pairs.
- `run_status.json`: run metadata.
