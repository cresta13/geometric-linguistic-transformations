# GLT-BUILD-01A: Corrected Composition Audit

Protocol fixed on 2026-09-15 before the corrected training run. This follows the
2026-09-11 pilot and is an exploratory protocol correction, not a preregistration
of the broader GLT-BUILD hypothesis.

Implementation: [audit runner](../../../scripts/run_glt_build_01_composition_audit.py)
and [regression tests](../../../scripts/test_glt_build_01_composition_audit.py).
The runner reuses the [original pilot model and grammar](../../../scripts/run_glt_build_01_explicit_operator_smoke.py).

## Question And Scope

Can the existing tiny model execute known operations on every held-out scene,
and can it combine operations through direct commands or successive generated
sentences? Does the measured geometry explain more than random initialization,
token/position embeddings, or a constant target prediction?

The architecture and training examples remain those of the pilot: a decoder-only
Transformer, two layers, four heads, width 64, dropout zero, 360 training scenes,
3600 examples, seeds 0/1/2, AdamW learning rate 0.003, weight decay 0.01, batch 64,
800 steps, and observations at 0/100/250/500/800. Ridge alpha stays 1.0. CPU threads
are limited to two and inference batches to 32 for this computer's memory budget.
No hyperparameter is selected using corrected evaluation results.

## Source Coverage

Evaluate all 88 held-out scenes for every operation in each test family:

| Family | Operation sequences per scene | Rows per seed/checkpoint |
|---|---:|---:|
| Single operation | 5 | 440 |
| Seen ordered pair | 4 | 352 |
| Reversed seen pair | 4 | 352 |
| Unseen pair | 3 | 264 |

Training sanity checks use two scenes from each of the eight subjects and all ten
taught sequences, for 160 rows. The held-out set is exhaustive, not artificially
equalized between subjects; its subject/verb/object counts are saved. All words
are known and the grammar is fixed. This is not a held-out vocabulary or template
experiment. No identical operation/source input crosses the training/test split;
individual endpoints can recur elsewhere, especially under role swap.

## Actual Composition

AB means A first, then B. For TN, TQ, NQ, VQ (seen pairs) and NV, TV, SQ (unseen):

1. Generate direct AB and BA commands from the original source.
2. Generate A(source), then feed that exact output into B; repeat B then A.
3. Separately feed the correct intermediate sentence into the second operation.
4. Record exact final-target correctness and first-step correctness, both final
   strings, order agreement, and the proportion where both orders are correct.

Never repair an incorrect intermediate before the sequential condition. The
correct-intermediate condition is explicitly labeled oracle. Since training
inputs are only active affirmative present statements, many second-stage inputs
are out of distribution. Failure there does not by itself isolate an algebraic
defect. These are text-mediated composition tests, not sequential hidden-matrix
interventions.

All five generator operations commute. Equality of generated reference endpoints
is a software invariant, not evidence of learned commutativity. Agreement between
two incorrect model outputs is reported separately from correct execution.

## Geometry And Causal Checks

At each observation save token-plus-position embeddings (layer 0) and both hidden
layers, with two predefined readouts: last prompt separator and mean of sentence
content tokens. Prefix tokens, separators, and padding are excluded from that
mean. These readouts do not completely eliminate lexical or positional effects.

Compare identity, training target mean, additive, linear, affine, wrong additive,
norm-matched random additive, and shuffled-training-pair linear/affine predictors.
Shuffling is within operation, fitted on training scenes only, and fixed across
checkpoints/readouts within a seed. One permutation is a diagnostic, not a null
distribution or significance test.

Report raw cosine, target-centered cosine (training target mean subtracted), and
RMSE relative to the error of the training target mean. Relative RMSE below 1 is
better than that constant prediction. If target variance or centered vector norms
are zero, the associated metric is undefined and remains blank. Checkpoint-zero
results must be shown alongside final results; raw cosine alone cannot establish
learning or emergence. Operator fits use sentence-copy prompts, including
transformed input forms absent from task training.

Retain the original intervention recipe: last layer, gain 1, mean target-minus-
source vector added at the last token on every full-prefix generation pass. Run
target/no/wrong/negative/random-vector conditions on all 88 sources per operation.
This is one recipe, with no layer or gain search.

## Saved Artifacts And Uncertainty

- Model weights, AdamW state, Torch RNG, batch sampler RNG and config at every
  observation, including initialization.
- Both sentence readouts for every layer and indexed sentence, in compressed NPZ.
  These are not full per-token activation dumps.
- CSV outputs, intermediate generated strings, source coverage, source-code
  hashes, exact command configuration, and SHA-256 manifests for binary artifacts.
- Rates with source/seed/row counts, seed standard deviation, and 95% percentile
  intervals from 1000 crossed scene/seed bootstrap draws. Resample scenes and
  seeds, never treat dependent generations as independent trials. Three seeds
  and boundary rates can give narrow or degenerate intervals; this does not
  establish population certainty.

Binary snapshots remain local and are reproducible using the recorded command;
tables, figures, metadata, manifests and summaries belong in the repository.

## Run And Stopping Rule

```powershell
.\.venv\Scripts\python.exe scripts/run_glt_build_01_composition_audit.py --out-dir results/experiments/glt_build_01_composition_audit_20260915_results --seeds 0,1,2 --steps 800 --checkpoints 0,100,250,500,800 --d-model 64 --layers 2 --heads 4 --batch-size 64 --eval-batch-size 32 --threads 2
```

Stop after these three seeds and the declared observations. Report coverage,
composition errors, geometry controls and intervention outcomes, even when null.
Do not add a layer sweep, more models, noncommuting generators, or new training
curricula to this run. Any further training-support change requires a separate
protocol informed by these results.
