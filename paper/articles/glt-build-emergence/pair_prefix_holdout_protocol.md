# GLT-BUILD-01A: Pair-Prefix Holdout

This experiment is declared after the completed input-support audit. Its question
is narrower:

> Can the model execute a new two-operation command prefix when it has seen every
> operation separately, but has never seen two operation tokens together during
> training?

The prior audit showed that expanded source-state teaching rescues sequential text
execution. That result leaves open whether the model learned reusable operations
or simply learned the joint prefixes that were present in training.

## Arms

| Arm | Training command prefixes | Source-state variants |
|---|---|---|
| `expanded_pairs` | identity, five singles, TN/TQ/NQ/VQ | all 16 variants for identity/singles; base form for pairs |
| `single_only` | identity and five singles only | all 16 variants |

The `single_only` arm contains no two-operation command prefix. Both arms use the
same 180 training groups and 88 held-out directed scenes from the input-support
split, the same vocabulary, two-layer/four-head width-64 Transformer, AdamW setup,
batch size 64, padded length 32, three seeds, and 1600 updates at checkpoints
0/400/800/1600. The arms have different command alphabets, so their random draws
are not claimed to be paired. Initial seeds and architecture are matched.

Training pools contain 36,000 rows for `expanded_pairs` and 34,560 rows for
`single_only`; rows are sampled rather than exhausted. The evaluation uses all
held-out scenes and retains direct outputs, generated-intermediate sequential
outputs and oracle-intermediate outputs.

## Primary Test

The primary metric is exact correctness for both sequential orders, averaged over
TN, TQ, NQ, VQ, NV and TV. SQ remains a separate positive-control pair. The direct
pair table is the key prefix test: the `single_only` arm is evaluated on all taught
and unseen two-operation prefixes despite having seen none during training.

Interpretation:

- high sequential scores in `single_only` with low direct-pair scores: learned
  primitive reuse through text-mediated steps, without command-prefix composition;
- high direct unseen-pair scores in `single_only`: evidence that the model combines
  operation tokens compositionally under this controlled grammar;
- failure of nonbase single operations would make a negative composition result
  inconclusive, so primitive competence is reported separately.

All generator operations commute. This is a command-composition test, not a
noncommutative or Lie-algebra test. No latent algebra claim follows from a positive
result.

## Run

```powershell
.\.venv\Scripts\python.exe scripts/run_glt_build_01_pair_prefix_holdout.py --out-dir results/experiments/glt_build_01_pair_prefix_holdout_20261008_results --seeds 0,1,2 --steps 1600 --checkpoints 0,400,800,1600 --threads 2 --eval-batch-size 32
```

Stop after the six model runs and declared checkpoints. Validate exact-match
labels, unique keys, complete row counts and checkpoint hashes. Do not add more
operation pairs, a layer sweep or another training budget to this run.
