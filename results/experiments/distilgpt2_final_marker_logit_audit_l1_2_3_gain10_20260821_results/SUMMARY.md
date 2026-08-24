# DistilGPT-2 Final-Marker Logit Audit

Date: 2026-08-21

This run repeats the GLT-STEER final-marker logit audit on `distilgpt2` to test whether the GPT-2 Final Marker Hypothesis transfers to a smaller distilled model.

Configuration:

- model: `distilgpt2`
- layers: `1,2,3`
- gain: `1.0`
- target markers: question `?`, exclamation `!`, ellipsis `...`
- prompt styles: `copy_sentence`, `same_sentence`
- sources: `12` in-template plus `12` hard out-of-template sources
- controls: `none`, `target`, `wrong_marker`, `random_norm`, `negative_target`
- rows: `2160` generated sequences and `51794` logged generation steps

Headline aggregate:

| target | control | mean marker rate | max marker rate | mean best marker prob | median best rank | top-1-any rate |
|---|---|---:|---:|---:|---:|---:|
| `?` | `none` | `0.0000` | `0.0000` | `0.0002` | `32.0` | `0.0000` |
| `?` | `target` | `0.2986` | `0.7500` | `0.2876` | `2.0` | `0.2986` |
| `!` | `none` | `0.0000` | `0.0000` | `0.0004` | `28.75` | `0.0000` |
| `!` | `target` | `0.7778` | `1.0000` | `0.7344` | `1.0` | `0.7778` |
| `...` | `none` | `0.0000` | `0.0000` | `0.0007` | `11.75` | `0.0000` |
| `...` | `target` | `0.5139` | `1.0000` | `0.4271` | `1.5` | `0.5139` |

Layer and marker sensitivity:

- Exclamation is the cleanest transferred marker. Target exclamation reaches `1.0000` marker rate on several layer/prompt/source cells.
- Ellipsis is strong mainly at layer `2`, where target ellipsis reaches `1.0000` across the tested source/prompt cells.
- Question is weaker and more unstable. The best question cells reach `0.7500`, but the aggregate target question rate is only `0.2986`.
- No-steering marker rates remain `0.0000` for all three targets, so the prompt-only explanation is still ruled out.

Interpretation:

This is a positive but model-dependent transfer result. DistilGPT-2 partially supports the Final Marker Hypothesis: target steering moves final-marker logits and can produce the intended marker, especially for `!` and layer-2 `...`. It does not replicate GPT-2 strength uniformly. The question marker is substantially weaker and more layer/prompt sensitive.

The result should be reported as **final-marker steering transfers to DistilGPT-2 with strong layer and marker dependence**, not as a full GPT-2 replication.

Files:

- `csv/final_marker_logit_summary.csv`
- `csv/final_marker_logit_sequence_raw.csv`
- `csv/final_marker_logit_steps.csv`
- `csv/final_marker_logit_sources.csv`
- `csv/final_marker_logit_training_pairs.csv`
- `csv/final_marker_logit_token_ids.json`
- `run_status.json`
