# GPT-2 negation copy-prompt layer sweep

This run tests whether the weak negation-steering result was caused by injecting the vector into the wrong GPT-2 layer.

## Setup

- Script: `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- Model: `gpt2`
- Transformation: `negation`
- Layers: `0-11`
- Gain: `0.75`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Sources: `40` in-template test sources and `40` out-of-template freeform sources
- Raw generations: `14400`

## Main result

The layer sweep does not find a clean negation-steering layer.

Best target row:

| source set | prompt style | target marker rate | target-and-preserved rate | rows |
|---|---|---:|---:|---:|
| `out_of_template_freeform` | `repeat_sentence` | `0.2104` | `0.1729` | `480` |

Maximum rates by control:

| control | max target marker rate | max target-and-preserved rate |
|---|---:|---:|
| `target` | `0.2104` | `0.1729` |
| `wrong_class` | `0.1604` | `0.1229` |
| `random_norm` | `0.1417` | `0.1083` |
| `none` | `0.1250` | `0.1000` |
| `negative_target` | `0.1188` | `0.0917` |

## Interpretation

The result remains a boundary result. Searching all GPT-2 layers improves the best target row relative to the previous compact negation run, but the separation from controls is small. This does not support the claim that the current single-vector copy-prompt steering recipe produces clean negation edits.

Combined with the delta-coherence diagnostic, this suggests that the negation failure is not only a layer-choice issue. Negation appears harder both geometrically and generatively: its deltas are less coherent than question deltas, and the model must insert a marker such as `not` inside the sentence rather than append a final punctuation mark.

## Files

- `csv/transformation_copy_prompt_summary.csv`: aggregate layer-sweep rates.
- `csv/transformation_copy_prompt_raw.csv`: all generated outputs.
- `csv/transformation_copy_prompt_sources.csv`: source sentences.
- `run_status.json`: run metadata.
