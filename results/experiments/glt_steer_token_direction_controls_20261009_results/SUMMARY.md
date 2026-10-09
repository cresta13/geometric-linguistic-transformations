# GLT-STEER Token-Direction Controls

Completed GPT-2 specificity audit for the final-marker steering effect.

## Protocol

- Model: `gpt2`
- Target: question marker `?`
- Layer: `2`
- Gain: `0.75`
- Sources: `40` structurally varied held-out sentences
- Prompt: `same_sentence`
- Rows: `360`
- Controls: learned target delta, shuffled-pair delta, target-token embedding, other punctuation, random word, random norm, and negative target delta.

The token controls include both the raw token embedding and a version rescaled to the norm of the learned target delta. The rescaled comparison is the relevant capacity-matched control.

## Results

| control | question marker rate | marker+preserved rate | content preserved rate |
|---|---:|---:|---:|
| `none` | `0.0000` | `0.0000` | `0.2000` |
| `target_delta` | `0.3750` | `0.0500` | `0.1750` |
| `shuffled_pair_delta` | `0.0500` | `0.0250` | `0.1750` |
| `target_token_embedding_raw` | `0.0000` | `0.0000` | `0.2000` |
| `target_token_embedding_normmatched` | `0.3750` | `0.0750` | `0.2000` |
| `other_punctuation_normmatched` | `0.0000` | `0.0000` | `0.0250` |
| `random_word_normmatched` | `0.0000` | `0.0000` | `0.0000` |
| `random_norm` | `0.0000` | `0.0000` | `0.1000` |
| `negative_target_delta` | `0.0000` | `0.0000` | `0.0750` |

## Interpretation

The learned target delta produces a real final-marker intervention under this protocol, and the shuffled-pair delta is much weaker. However, the norm-matched embedding of the target token `?` reproduces the same question-marker rate as the learned delta. Therefore this audit does **not** support the stronger claim that the sentence-pair transformation delta is necessary or uniquely informative for question-marker induction.

The bounded conclusion is narrower: GPT-2 can be pushed toward a final punctuation marker by an activation-space direction, but the current result is compatible with a token-direction or output-form mechanism. The learned delta should not be presented as a uniquely linguistic transformation vector without additional controls or a task where token embedding directions cannot explain the behavior.

## Files

- `csv/glt_steer_token_direction_controls_raw.csv`
- `csv/glt_steer_token_direction_controls_summary.csv`
- `csv/token_control_ids.json`
- `scripts/run_glt_steer_token_direction_controls.py`
