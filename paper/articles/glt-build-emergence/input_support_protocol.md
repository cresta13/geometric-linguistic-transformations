# GLT-BUILD-01A: Paired Training-Input Support Experiment

Declared on 2026-09-15 before the research run. This is a bounded follow-up to
[the completed composition audit](../../../results/experiments/glt_build_01_composition_audit_20260915_results/SUMMARY.md).
Implementation: [runner](../../../scripts/run_glt_build_01_input_support.py) and
[tests](../../../scripts/test_glt_build_01_input_support.py).

## Question

Does teaching single operations on transformed source sentences rescue the
text-mediated composition failures observed when all training inputs were active,
affirmative, present-tense statements? Keep actual sequential execution separate
from a novel pair of operator tokens in one prompt.

## Paired Intervention

| Property | base_only | expanded_sources |
|---|---|---|
| Identity and single-operation sources | Base form only | All 16 binary tense/polarity/voice/mood combinations |
| Taught two-command sequences | TN, TQ, NQ, VQ | Same |
| Sources for taught two-command sequences | Base form only | Base form only |
| Unseen pair command tokens | NV, TV, SQ and their reversals absent | Same |
| Available unique training examples | 3600 | 36000 |

Draw scene, command and a variant index with the same RNG schedule in both arms.
Each of ten commands has probability 1/10; the variant index is uniform on 0..15.
The control ignores the variant. The treatment uses it only for identity and the
five single operations. Pair commands always ignore it. Thus 60% of draws target
identity/single operations and 40% taught pairs in expectation, with identical
actual command counts across paired arms. The expanded pool is not sampled
uniformly over its 36000 rows: keeping command exposure paired is intentional.

Both arms start from identical weights for each seed. The model is a two-layer,
four-head, width-64 decoder-only Transformer, dropout zero, vocabulary 50, maximum
position table 64. Optimizer: AdamW, learning rate 0.003, weight decay 0.01, gradient
clip 1.0, batch 64. Train for **1600 updates**, recording **0/400/800/1600**. The
1600-step endpoint is primary; 800 is a fixed intermediate budget comparison.
Seeds: **0, 1, 2**. All six models run sequentially on CPU, two threads, inference
batch 32, low process priority.

Every training batch is padded to length 32 in both arms, with future causal
masking and padding labels ignored. This matches update counts, examples seen and
the model's padded computational shapes. Longer targets still expose more
supervised tokens in the treatment; token counts are recorded and are **not**
claimed to be matched. Available unique examples and effective epoch counts also
differ as an inherent part of this support expansion.

## Leakage-Resistant Split

The independent split unit is an unordered participant pair plus a verb. Swapping
roles or changing tense, polarity, voice or mood cannot change its split.

There are 224 such groups. Using `random.Random(1731)`, sample 6 of the 28 groups
for each of the first four verbs and 5 for each of the last four. This yields
**44 test groups / 88 directed scenes**, and **180 train groups / 360 directed
scenes**. All source-state forms and all transformation endpoints stay in their
group. Training/test surface-form overlap is checked before launch.

The split and deterministic vocabulary ordering differ from the previous audit.
Therefore compare the two new paired arms, not new scores against the old pilot
as if training examples or initial parameters were identical. Words and grammar
are shared; this is not lexical or natural-corpus generalization.

## Evaluation

At each checkpoint use every held-out directed scene:

- 5 single operations on the base form: 440 rows.
- 5 single operations on 15 nonbase forms: 6600 rows.
- Identity/copy on 15 nonbase forms: 1320 rows.
- Seen/reversed seen/unseen direct pairs: 352/352/264 rows.
- Direct AB/BA, generated-intermediate sequential AB/BA, and correct-intermediate
  oracle AB/BA for all seven pairs: 616 wide composition rows, including output
  strings, first-step correctness, order agreement and both-orders-correct.

The sequential second step receives the actual first output, without correction.
The oracle condition remains separate. All external operations commute; two wrong
outputs agreeing is not successful composition or learned commutativity.

## Primary Outcome And Fixed Interpretation Rule

Primary outcome: the proportion for which **both sequential orders are exactly
correct**, averaged over the six previously failing pairs TN/TQ/NQ/VQ/NV/TV.
Keep SQ as a separate positive-control pair so it cannot inflate this endpoint.

Call the result a clean support rescue under this protocol only if:

1. At step 1600, treatment minus control improves the primary rate by at least
   0.20, and its paired 95% bootstrap lower bound is above zero.
2. Both arms retain at least 95% base-source single-operation accuracy.
3. The treatment reaches at least 95% nonbase-source single-operation accuracy.

These are declared decision thresholds, not universal scientific cutoffs. Report
all rates even when the rule is not met. Failure to learn the expanded primitive
task makes a negative composition result inconclusive about sufficient training
support at a larger budget. Success in sequential execution with failure on novel
pair commands distinguishes primitive reuse from command-prefix generalization.
Neither result establishes a latent algebra or a successful hidden-state edit.

Use paired treatment/control differences within semantic group and seed, then
1000 crossed group/seed bootstrap draws. Keep role reversals, variants and all
operations inside their groups. Tables report 44 semantic groups, three seeds,
dependent row counts and seed variability. Three seeds and boundary rates limit
the resolution of the intervals, including degenerate empirical intervals.

## Saved Outputs And Stopping Rule

Save source pools, evaluation examples, split coverage, generated/intermediate
strings, training-token counts, paired weight/sampling hashes, and automatic
summaries/figures. Save local weights, optimizer/RNG state and two hidden readouts
at each observation. The fixed hidden panel contains all 16 forms of every test
scene plus two training scenes per subject. It supports later inspection; no
new geometry-fit or steering sweep is part of this experiment. Binary snapshots
remain local, with sizes and SHA-256 hashes in a compact public manifest.

Stop after the six models and four observations. Validate complete row counts,
labels, paired initialization/sampling and all binary hashes before marking the
run finished. On failure, record an error and retain completed checkpoints. Do
not automatically extend training, add seeds, or launch another experiment.

```powershell
.\.venv\Scripts\python.exe scripts/run_glt_build_01_input_support.py --out-dir results/experiments/glt_build_01_input_support_20260915_results --seeds 0,1,2 --steps 1600 --checkpoints 0,400,800,1600 --threads 2 --eval-batch-size 32
```
