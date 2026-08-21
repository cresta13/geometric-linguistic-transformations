# GPT-2 Final-Marker Logit Audit

Date: 2026-08-10

This run tests whether GLT-STEER final-marker vectors actually move the target marker token into the model's next-token logits during generation, rather than relying on a high no-steering base rate.

Configuration:

- model: `gpt2`
- layers: `2,3`
- gain: `0.75`
- target markers: question `?`, exclamation `!`, ellipsis `...`
- prompt styles: `copy_sentence`, `same_sentence`
- sources: `12` in-template plus `12` hard out-of-template sources
- controls: `none`, `target`, `wrong_marker`, `random_norm`, `negative_target`
- rows: `1440` generated sequences and `34343` logged generation steps

Headline aggregate:

| target | control | mean marker rate | mean best marker prob | median best rank | top-1-any rate |
|---|---|---:|---:|---:|---:|
| `?` | `none` | `0.0000` | `0.0015` | `28.0` | `0.0000` |
| `?` | `target` | `0.8542` | `0.7537` | `1.0` | `0.8542` |
| `!` | `none` | `0.0000` | `0.0010` | `20.0` | `0.0000` |
| `!` | `target` | `0.9063` | `0.7718` | `1.0` | `0.9063` |
| `...` | `none` | `0.0000` | `0.0016` | `11.75` | `0.0000` |
| `...` | `target` | `0.8750` | `0.7403` | `1.0` | `0.8750` |

Target-vs-control summary:

- The no-steering condition produces no target markers for all three final-marker classes.
- Target steering moves the intended marker to rank `1` in most sequences and produces target markers at `0.8542-0.9063` aggregate rates.
- Random-norm and negative-target controls produce no target markers in the aggregate marker-rate table.
- Wrong-marker vectors do not produce the target marker, although they can raise another final-marker token. This is expected and supports the class-specific final-marker interpretation.

Interpretation:

The result supports the Final Marker Hypothesis at the logit level. The marker tokens are not simply already being emitted by the copy-like prompts: under `none`, marker rates are `0.0000`. Under target steering, the corresponding marker token is repeatedly moved into the top logit rank during generation.

This remains a final-surface-marker result. It does not show that the same simple last-token residual intervention works for lexical or sentence-internal transformations.

Files:

- `csv/final_marker_logit_summary.csv`
- `csv/final_marker_logit_sequence_raw.csv`
- `csv/final_marker_logit_steps.csv`
- `csv/final_marker_logit_sources.csv`
- `csv/final_marker_logit_training_pairs.csv`
- `csv/final_marker_logit_token_ids.json`
- `run_status.json`
