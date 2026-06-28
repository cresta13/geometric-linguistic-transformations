# Partial run note

This directory preserves the completed `intfloat/multilingual-e5-large-instruct`
result from a multi-model exploratory run.

The run terminated during the next model (`Alibaba-NLP/gte-multilingual-base`)
without producing a final `run_status.json`; therefore only files with the
`intfloat__multilingual_e5_large_instruct` suffix and the aggregate CSVs derived
from that completed checkpoint should be interpreted.

`Qwen/Qwen3-Embedding-0.6B` was rerun separately after a dtype fix and is stored
in `results/experiments/lie_multilingual_qwen3_embedding_48t_2000null_results/`.
