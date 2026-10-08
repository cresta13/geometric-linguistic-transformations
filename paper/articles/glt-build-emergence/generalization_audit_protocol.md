# GLT-BUILD-03: Template and Lexical Generalization Audit

This protocol stress-tests the bounded order-sensitive composition result from
GLT-BUILD-02. It keeps the same tiny decoder-only architecture, operations,
scene split, seeds, and evaluation budget, but changes the surface support
between training and evaluation.

## Question

Does sequential execution of the noncommuting R/M pair survive a new
surface-template combination and held-out entity/verb scene groups, or was the
previous result tied to the training surface?

## Operations

R swaps subject and object. M toggles a mark on the current subject. The
ordered routes RM and MR should therefore produce distinct target states. T
and N remain the commuting control pair.

## Training and evaluation support

Identity and single-operation examples are trained with two templates:

    Mira opens the gate .
    the Mira opens the gate today .

Evaluation uses the held-out combination:

    Mira opens the gate today .

The evaluation scene groups are disjoint from the training scene groups under
the existing role-swap-closed split. Pair prefixes are exposed only for RM
and TN in the pair_exposed arm; single_only receives no pair prefixes.
The primary comparison remains sequential execution, not direct pair-prefix
prediction.

## Fixed configuration

- two layers, four heads, width 64, batch size 64;
- three seeds: 0, 1, 2;
- 1600 updates;
- checkpoints: 0, 400, 800, 1600;
- CPU execution with two Torch threads;
- evaluation batch size 32;
- same held-out scene-group split as GLT-BUILD-02.

## Primary endpoints

1. sequential RM and MR exact-match rates;
2. whether sequential RM and MR outputs remain distinct;
3. commuting TN/NT correctness and order invariance;
4. identity and single-operation correctness on the held-out surface/scene set.

The result is interpreted as generalization only if labels match saved strings,
all expected rows are present, duplicate keys are absent, and checkpoint
hashes validate. This is still a finite synthetic behavioral test. A positive
result would not establish a Lie algebra, a universal operator, or natural
language generalization.

## Reproduction

    .\.venv\Scripts\python.exe scripts/run_glt_build_03_generalization_audit.py --out-dir results/experiments/glt_build_03_template_lexical_generalization_20261009_results --seeds 0,1,2 --steps 1600 --checkpoints 0,400,800,1600 --threads 2 --eval-batch-size 32
