# GPT-2 steering delta coherence diagnostic

This diagnostic tests whether the successful question-steering result and the weak negation-steering result differ in the internal coherence of their transformation deltas.

## Setup

- Script: `scripts/analyze_gpt2_steering_delta_coherence.py`
- Model: `gpt2`
- Classes: `question`, `negation`
- Layers: `0-11`
- Rows per class: `100`
- Delta definition: last-token hidden state of target sentence minus last-token hidden state of source sentence.

## Main result

Question deltas are much more internally coherent than negation deltas across all GPT-2 layers.

| layer | question mean pairwise cosine | negation mean pairwise cosine | gap |
|---:|---:|---:|---:|
| 0 | 0.9762 | 0.4786 | 0.4976 |
| 1 | 0.9814 | 0.5706 | 0.4108 |
| 2 | 0.9693 | 0.5621 | 0.4072 |
| 3 | 0.9365 | 0.5665 | 0.3700 |
| 4 | 0.9006 | 0.5425 | 0.3581 |
| 5 | 0.8748 | 0.4835 | 0.3913 |
| 6 | 0.8369 | 0.4714 | 0.3655 |
| 7 | 0.8204 | 0.4428 | 0.3776 |
| 8 | 0.8076 | 0.4293 | 0.3783 |
| 9 | 0.7896 | 0.4109 | 0.3787 |
| 10 | 0.7932 | 0.3884 | 0.4049 |
| 11 | 0.9586 | 0.1373 | 0.8213 |

The layers used in the focused steering runs (`2` and `3`) show the same pattern:

- layer `2`: question `0.9693`, negation `0.5621`
- layer `3`: question `0.9365`, negation `0.5665`

## Interpretation

This supports the geometric-coherence explanation for why question steering works more cleanly than negation steering under the current recipe.

Question formation has a highly clustered delta direction in GPT-2 hidden space, so its centroid vector is a strong and stable intervention direction. Negation deltas are much more dispersed, so the class centroid is a blurrier object and is less likely to work as a clean activation-steering vector.

This does not rule out the autoregressive-marker explanation: question formation also has an easy final surface marker (`?`), while negation often requires inserting a token such as `not` inside the sentence. The current result says that the two explanations are compatible: question is both geometrically cleaner and easier to express at generation time.

## Files

- `csv/delta_coherence_by_layer_class.csv`: layer/class coherence metrics.
- `csv/question_negation_layerwise_contrast.csv`: question-minus-negation coherence gap by layer.
- `csv/delta_between_class_cosine.csv`: between-class centroid comparisons.
- `csv/coherence_pairs.csv`: sampled pairwise cosine values used for the aggregate statistics.
- `run_status.json`: run metadata.
