# GLT-BUILD-01: Emergence of a Known Transformation Algebra in a Tiny Synthetic-Language Transformer

## Status

Planned experiment. No results are claimed in this document.

## Motivation

Earlier GLT tracks mostly inspect already-trained models. This has produced useful but bounded results: GLT-STEER finds causal final-marker steering in GPT-2-style residual streams, while GLT-DV, GLT-SPOT, GLT-MOLT, and GLT-XFER expose representation and operator diagnostics in frozen embedding spaces.

The limitation is structural. In pretrained models, the relevant representations were shaped by unknown data, unknown optimization history, and many interacting features. A simple transformation may have existed earlier in training and later become distributed, mixed, nonlinear, or context-dependent.

GLT-BUILD changes the setup:

> Instead of only looking for transformation geometry after training, train a small controlled model and observe how the geometry appears.

## Main Research Question

How do linguistic transformations form inside a neural network during training, and what is the minimal mathematical structure needed to describe them?

The experiment tracks the path:

```text
random initialization
-> example memorization
-> feature emergence
-> transformation representation
-> composition behavior
-> held-out combination generalization
-> stabilization or collapse of structure
```

## Model

Initial model family:

```text
architecture: decoder-only Transformer
layers: 2
heads: 2-4
d_model: 32 or 64
vocabulary: small synthetic vocabulary
seeds: at least 10 for the promoted run
checkpoints: fixed training intervals
```

The first implementation may use a smaller smoke-test setting before the promoted run. Smoke-test results must not be treated as final evidence.

## Synthetic Language

Each sentence is generated from a known latent state:

```text
subject  = Mira
action   = open
object   = gate
tense    = present
polarity = affirmative
voice    = active
mood     = statement
```

Example surface sentence:

```text
Mira opens the gate.
```

Related variants:

```text
Mira opened the gate.
Mira will open the gate.
Mira does not open the gate.
The gate is opened by Mira.
Does Mira open the gate?
Mira opens the gate!
```

Because the generator is known, every sentence pair has a known latent transformation.

## Transformations

Initial discrete operations:

| Symbol | Operation |
|---|---|
| `T` | tense change |
| `N` | negation toggle |
| `V` | active/passive voice toggle |
| `Q` | statement/question mood toggle |
| `S` | semantic role swap |

Expected laws include:

```text
N^2 = I
V^2 = I
NT = TN
AB != BA for order-sensitive operation pairs
```

These are initially discrete transformations. The correct first mathematical objects are groups, semigroups, monoids, group actions, and matrix/operator representations. Lie-algebra analysis is reserved for later continuous or quasi-continuous features.

## Training Regimes

### GLT-BUILD-01A: Explicit Operator Tokens

The model receives an operation token and a source sentence, then generates the transformed sentence:

```text
<NEGATE> Mira opens the gate.
-> Mira does not open the gate.

<PAST> Mira opens the gate.
-> Mira opened the gate.
```

Purpose: test how a model encodes known operations and their compositions when the transformation labels are explicit.

### GLT-BUILD-01B: Implicit Transformations

The model is trained with next-token prediction on a synthetic corpus that contains related forms of the same latent scenes, without explicit operator labels.

Purpose: test whether transformation structure appears without named operation tokens.

The explicit-token regime should be built first. The implicit regime is a second-stage experiment, not a prerequisite for the first smoke test.

## Held-Out Structure

The dataset must include controlled exclusions:

- held-out lexemes;
- held-out templates;
- held-out operation compositions;
- held-out lexeme-by-composition combinations.

Example:

```text
train: PAST
train: NEGATION
train: PAST + NEGATION for some verbs
test:  PAST + NEGATION for different held-out verbs
```

If the model handles held-out combinations, this is evidence for compositional generalization rather than pure memorization.

## Checkpoint Observations

At fixed training checkpoints, store hidden states:

```text
h_l^t(x)
```

where:

- `t` is the training checkpoint;
- `l` is the layer;
- `x` is the sentence or operator-conditioned input.

Measured questions:

- when transformations first become linearly separable;
- which layer represents each operation most clearly;
- whether effective dimensionality shrinks or grows during training;
- whether representation structure appears before or after behavioral generalization;
- whether the same structure appears across random seeds;
- whether a structure is stable or transient.

## Representation Models To Compare

For each operation, test a ladder of increasing complexity:

### Additive Vector

```text
h(Tx) ~= h(x) + v_T
```

### Linear Operator

```text
h(Tx) ~= A_T h(x)
```

### Affine Operator

```text
h(Tx) ~= A_T h(x) + b_T
```

### Low-Rank or Context-Conditioned Operator

```text
h(Tx) ~= A_T(x)h(x) + b_T(x)
```

### Nonlinear Map

```text
h(Tx) ~= f_T(h(x))
```

The promoted result is the minimal complexity needed to explain each transformation, not the discovery of a vector by default.

## Composition Tests

For learned matrix or affine operators, test:

```text
A_(AB) ~= A_A A_B
A_I ~= I
A_(A^-1) ~= A_A^-1
A_A A_B ~= A_B A_A      for independent operations
A_A A_B != A_B A_A      for order-sensitive operation pairs
```

For additive deltas, test whether:

```text
v_(AB) ~= v_A + v_B
```

and whether the approximation breaks differently for independent versus order-sensitive operation pairs.

## Causal Interventions

A representation candidate is stronger if it changes behavior, not only if it predicts hidden states.

For each candidate operation object:

1. observe whether `h(Tx)` matches the proposed vector/operator model;
2. intervene on `h(x)` with the candidate vector/operator;
3. decode or continue generation;
4. measure whether the output changes toward the target transformation.

This connects GLT-BUILD to GLT-STEER, but under a cleaner setting where the data generator and training history are known.

## Required Controls

Every promoted result must include:

- no-intervention control;
- random norm-matched vector/operator controls;
- negative and inverse controls;
- wrong-transformation controls;
- shuffled-label controls;
- source-only, target-only, and endpoint-only baselines;
- token-only punctuation vectors where final markers are tested;
- input-embedding and unembedding-direction baselines;
- arbitrary terminal-token controls;
- exact final-position marker checks, not marker-anywhere checks;
- exact and semantic content-preservation checks;
- repetition and malformed-generation checks;
- multiple random seeds;
- multiple model sizes or architectures for promoted claims;
- held-out lexemes, templates, and compositions;
- cluster-aware confidence intervals over independent source scenes.

## Initial Deliverables

1. Synthetic language generator.
2. Tiny decoder-only Transformer training script.
3. Checkpointed hidden-state extraction.
4. Behavioral generalization table.
5. Representation separability over time.
6. Additive/linear/affine/operator-fit comparison over time.
7. Composition-law diagnostics over time.
8. Causal intervention audit.
9. `RUN_SUMMARY.md` for the result folder.
10. Research-diary entry before interpreting outcomes.

## Stopping Rule For GLT-BUILD-01

The first promoted GLT-BUILD result is complete when it can answer these questions:

1. Did the tiny model learn the explicit transformation task on held-out combinations?
2. At which checkpoint and layer did each operation first become recoverable?
3. Which representation model is sufficient for each operation: additive, linear, affine, low-rank/contextual, or nonlinear?
4. Do learned operation objects preserve the known composition laws better than matched nulls?
5. Does applying the discovered object causally change model behavior?

If the answer is negative, the result is still useful: it shows that even in a controlled synthetic setup, the model may solve the task through memorization or distributed/contextual mechanisms rather than a stable shared transformation geometry.

## Safe Claim Template

Positive version:

> In a controlled synthetic-language Transformer, some linguistic transformations become recoverable during training as compact representation-level objects. These objects appear at identifiable checkpoints/layers, partly preserve externally defined composition laws, and support causal interventions on held-out combinations.

Negative version:

> In a controlled synthetic-language Transformer, successful behavioral learning does not necessarily imply a stable additive or operator-level transformation geometry. This bounds the search for universal transformation vectors in pretrained models.

Both outcomes are scientifically useful.
