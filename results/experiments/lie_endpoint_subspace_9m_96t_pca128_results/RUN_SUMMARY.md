# 9-Model Endpoint-Subspace Residualization Audit

Completed: 2026-06-29

This run extends the endpoint-subspace residualization audit to nine multilingual embedding models, seven languages, 96 templates per language, and PCA dimension 128.

Models:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/LaBSE`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `distilbert-base-multilingual-cased`
- `intfloat/multilingual-e5-large-instruct`
- `Qwen/Qwen3-Embedding-0.6B`

Languages: English, Spanish, French, German, Russian, Chinese, Arabic.

## Main Result

All four tested triples remain below the exact signed-null baseline across all `63/63` model-language cells after removing endpoint-derived linear rowspaces for triple label, endpoint position, and cyclic sign.

Global mean ratios to exact signed-null:

| Triple | Raw | Remove all endpoint subspaces | Mean fraction below null |
| --- | ---: | ---: | ---: |
| `NQM` | `0.549` | `0.547` | `1.000` |
| `QMT` | `0.602` | `0.615` | `1.000` |
| `NQT` | `0.684` | `0.685` | `1.000` |
| `NMT` | `0.756` | `0.755` | `0.9997` |

## Endpoint Probe Check

Endpoint-derived information is present, but does not explain away the residualized signed-permutation signal under these linear controls.

Probe macro F1 summary:

| Probe | Mean macro F1 | Chance |
| --- | ---: | ---: |
| Triple label from single endpoint delta | `0.765` | `0.250` |
| Endpoint position from endpoint delta | `0.283` | `0.167` |
| Cyclic vs anticyclic from endpoint delta | `0.526` | `0.500` |

## Interpretation

This is the strongest current GLT-SPOT stress test against a simple linear endpoint-artifact explanation. It supports a robust controlled signed-permutation coherence signal across older and newer multilingual embedding models.

It is still not a proof of a formal Lie algebra. The remaining next step is to move from endpoint/delta residualization toward learned operator maps and stronger endpoint-balanced generation.

## Files

- Global residualization summary: `csv/subspace_residualized_global_summary.csv`
- Model/language summaries: `csv/subspace_residualized_summary_all_models.csv`
- Raw residualization rows: `csv/subspace_residualized_raw_all_models.csv`
- Endpoint probe summary: `csv/endpoint_subspace_probe_summary.csv`
- Figure: `figures/01_subspace_residualization_ratios.png`
