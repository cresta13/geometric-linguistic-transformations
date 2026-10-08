# GLT-BUILD-02: Order-Sensitive Composition

Completed exploratory run testing a genuinely noncommuting pair in a controlled synthetic language.

## Design

`R` swaps subject and object. `M` toggles a mark on the current subject. Both are involutions, but `R` and `M` do not commute: marking then swapping marks the original subject as object, while swapping then marking marks the original object as subject. `T` toggles tense and `N` toggles negation; they are an independent commuting control.

The `single_only` arm sees identity and all four single operations, but no pair prefix. The `pair_exposed` arm additionally sees RM and TN pair prefixes on base sources. Both arms use the same 360/88 split, 12 source-state variants for identity/singles, three seeds, two layers, four heads, width 64, batch 64 and 1600 updates.

## Final Behavioral Results

| arm | family | exact_match |
| --- | --- | --- |
| pair_exposed | identity | 1.0000 |
| pair_exposed | single | 1.0000 |
| single_only | identity | 1.0000 |
| single_only | single | 1.0000 |

| arm | pair | order_distinct | direct_ab_correct | direct_ba_correct | sequential_ab_correct | sequential_ba_correct | oracle_ab_correct | oracle_ba_correct | direct_order_distinct_output | sequential_order_distinct_output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_exposed | MR | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 1.0000 |
| pair_exposed | MT | 0.0000 | 0.0000 | 0.5909 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| pair_exposed | NR | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7121 | 0.0000 |
| pair_exposed | NT | 0.0000 | 0.3106 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6894 | 0.0000 |
| pair_exposed | RM | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 1.0000 |
| pair_exposed | RN | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7121 | 0.0000 |
| pair_exposed | TM | 0.0000 | 0.5909 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| pair_exposed | TN | 0.0000 | 1.0000 | 0.3106 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6894 | 0.0000 |
| single_only | MR | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8712 | 1.0000 |
| single_only | MT | 0.0000 | 0.0000 | 0.2386 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9205 | 0.0000 |
| single_only | NR | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 0.0000 |
| single_only | NT | 0.0000 | 0.1212 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| single_only | RM | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8712 | 1.0000 |
| single_only | RN | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9962 | 0.0000 |
| single_only | TM | 0.0000 | 0.2386 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9205 | 0.0000 |
| single_only | TN | 0.0000 | 0.0000 | 0.1212 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Decision

| arm | direct_ab_correct | direct_ba_correct | sequential_ab_correct | sequential_ba_correct |
| --- | --- | --- | --- | --- |
| pair_exposed | 0.3627 | 0.3627 | 1.0000 | 1.0000 |
| single_only | 0.0450 | 0.0450 | 1.0000 | 1.0000 |

| arm | sequential_ab_correct | sequential_ba_correct | sequential_order_distinct_output |
| --- | --- | --- | --- |
| pair_exposed | 1.0000 | 1.0000 | 1.0000 |
| single_only | 1.0000 | 1.0000 | 1.0000 |

| arm | sequential_ab_correct | sequential_ba_correct | sequential_order_distinct_output |
| --- | --- | --- | --- |
| pair_exposed | 1.0000 | 1.0000 | 0.0000 |
| single_only | 1.0000 | 1.0000 | 0.0000 |

The primary order-sensitive contrast is RM versus MR. A successful result requires both sequential orders to be correct and to produce distinct outputs, while the TN/NT control should remain order-invariant. Direct prefix accuracy is reported separately because a model can execute two single commands sequentially without constructing a new pair prefix.

Decision: **order_sensitive_composition_supported**.

## Limits

This is a discrete finite-state composition test. It does not establish a Lie algebra, continuous generators, or a universal latent operator. Rates are descriptive across three seeds and 44 held-out groups. The language is synthetic and the pair-exposed arm has a different command alphabet, so the arms are matched by seed and architecture but not claimed to have identical command sampling.

![Order-sensitive outcomes](figures/order_sensitive_outcomes.png)
