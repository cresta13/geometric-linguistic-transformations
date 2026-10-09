# GLT-STEER Source-Cluster Bootstrap

This derived audit recomputes uncertainty for the fixed-parameter confirmatory GLT-STEER run with the independent sentence source as the resampling unit. Prompt styles and layers remain repeated measurements within each source cluster; they are not treated as independent sources.

- Input: `results\experiments\glt_steer_confirmatory_fixed_params_20260825_results\csv\glt_steer_confirmatory_raw.csv`
- Bootstrap unit: `source_id`
- Replicates: `5000`
- Seed: `20261009`
- Intervals: percentile 95% cluster-bootstrap intervals.

The intervals are descriptive and do not repair the synthetic-template limitation. They provide a more conservative uncertainty check than row-level Wilson intervals for the confirmatory source set.

## Selected results

| model | target | control | sources | marker rate (95% cluster CI) | marker+preserved (95% cluster CI) |
|---|---|---|---:|---:|---:|
| `distilgpt2` | `ellipsis` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `ellipsis` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `ellipsis` | `random_norm` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `ellipsis` | `target` | 48 | 0.8333 [0.7500, 0.9097] | 0.1597 [0.0764, 0.2569] |
| `distilgpt2` | `ellipsis` | `wrong_marker` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `exclamation` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `exclamation` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `exclamation` | `random_norm` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `exclamation` | `target` | 48 | 0.8958 [0.8264, 0.9514] | 0.3472 [0.2431, 0.4653] |
| `distilgpt2` | `exclamation` | `wrong_marker` | 48 | 0.0139 [0.0000, 0.0347] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `question` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `question` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `question` | `random_norm` | 48 | 0.0069 [0.0000, 0.0208] | 0.0000 [0.0000, 0.0000] |
| `distilgpt2` | `question` | `target` | 48 | 0.6319 [0.5556, 0.7083] | 0.0556 [0.0208, 0.0972] |
| `distilgpt2` | `question` | `wrong_marker` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `ellipsis` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `ellipsis` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `ellipsis` | `random_norm` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `ellipsis` | `target` | 48 | 0.8438 [0.7604, 0.9132] | 0.3854 [0.2604, 0.5139] |
| `gpt2` | `ellipsis` | `wrong_marker` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `exclamation` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `exclamation` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `exclamation` | `random_norm` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `exclamation` | `target` | 48 | 0.6875 [0.5764, 0.7882] | 0.3958 [0.2743, 0.5208] |
| `gpt2` | `exclamation` | `wrong_marker` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `question` | `negative_target` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `question` | `none` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `question` | `random_norm` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `gpt2` | `question` | `target` | 48 | 0.6562 [0.5556, 0.7501] | 0.2396 [0.1458, 0.3368] |
| `gpt2` | `question` | `wrong_marker` | 48 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Interpretation

The target/control separation remains a source-level descriptive result, but the uncertainty is wider than a row-level interval would suggest because each source contributes multiple prompt and layer rows. This audit does not create new model evidence; it reports the dependence structure explicitly.

The current GLT-STEER claim remains bounded to final-marker activation steering. It does not establish semantic rewriting or a general linguistic operator.
