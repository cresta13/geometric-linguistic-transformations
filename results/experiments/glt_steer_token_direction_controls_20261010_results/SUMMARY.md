# GLT-STEER Token-Direction Controls

This GPT-2 audit tests whether final-marker steering is specific to the learned sentence-pair delta direction or can be reproduced by unrelated token and matched-norm directions.

- Model: `gpt2`
- Target: question marker `?`
- Layer: `2`
- Gain: `0.75`
- Training pool: the exact shuffled `120`-source pool used by the confirmatory protocol (`12` subjects)
- Sources: `40` structurally varied held-out sentences
- Prompt: `same_sentence`
- Controls: learned target delta, shuffled-pair delta, target-token embedding, other punctuation, random word, random norm, and negative target delta.

## Results

| control | question marker rate | marker+preserved rate | rows |
|---|---:|---:|---:|
| `negative_target_delta` | 0.0000 | 0.0000 | 40 |
| `none` | 0.0000 | 0.0000 | 40 |
| `other_punctuation_normmatched` | 0.0000 | 0.0000 | 40 |
| `random_norm` | 0.0000 | 0.0000 | 40 |
| `random_word_normmatched` | 0.0000 | 0.0000 | 40 |
| `shuffled_pair_delta` | 0.0750 | 0.0250 | 40 |
| `target_delta` | 0.3000 | 0.0500 | 40 |
| `target_token_embedding_normmatched` | 0.3250 | 0.0250 | 40 |
| `target_token_embedding_raw` | 0.0000 | 0.0000 | 40 |

## Interpretation

The target delta is the prespecified positive direction. The token and shuffled-pair controls are tests of the alternative explanation that any direction with a related norm, or a direction associated with a punctuation token, is sufficient. The training pool is identical to the fixed-parameter confirmatory protocol rather than the first unshuffled slice of the Cartesian product. This audit remains limited to one model, one layer, one marker, and one prompt protocol; it is a specificity control, not a general semantic-editing test.

The raw generations and aggregate table are retained so that the control can be inspected without rerunning inference.
