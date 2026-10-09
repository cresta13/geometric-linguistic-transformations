# GLT-BUILD-03: Template Control Audit

This post-hoc control evaluates the same final checkpoints on two templates seen during training and the suffix-only combination held out by the protocol.

## Behavior

| arm | template | family | exact_match |
| --- | --- | --- | --- |
| pair_exposed | article_suffix | identity | 1.0000 |
| pair_exposed | article_suffix | single | 1.0000 |
| pair_exposed | canonical | identity | 1.0000 |
| pair_exposed | canonical | single | 1.0000 |
| pair_exposed | suffix_only | identity | 0.0000 |
| pair_exposed | suffix_only | single | 0.0015 |
| single_only | article_suffix | identity | 1.0000 |
| single_only | article_suffix | single | 1.0000 |
| single_only | canonical | identity | 1.0000 |
| single_only | canonical | single | 1.0000 |
| single_only | suffix_only | identity | 0.0000 |
| single_only | suffix_only | single | 0.0065 |

## Sequential Composition

| arm | template | pair | order | exact_match |
| --- | --- | --- | --- | --- |
| pair_exposed | article_suffix | MR | ab | 1.0000 |
| pair_exposed | article_suffix | MR | ba | 1.0000 |
| pair_exposed | article_suffix | NT | ab | 1.0000 |
| pair_exposed | article_suffix | NT | ba | 1.0000 |
| pair_exposed | article_suffix | RM | ab | 1.0000 |
| pair_exposed | article_suffix | RM | ba | 1.0000 |
| pair_exposed | article_suffix | TN | ab | 1.0000 |
| pair_exposed | article_suffix | TN | ba | 1.0000 |
| pair_exposed | canonical | MR | ab | 1.0000 |
| pair_exposed | canonical | MR | ba | 1.0000 |
| pair_exposed | canonical | NT | ab | 1.0000 |
| pair_exposed | canonical | NT | ba | 1.0000 |
| pair_exposed | canonical | RM | ab | 1.0000 |
| pair_exposed | canonical | RM | ba | 1.0000 |
| pair_exposed | canonical | TN | ab | 1.0000 |
| pair_exposed | canonical | TN | ba | 1.0000 |
| pair_exposed | suffix_only | MR | ab | 0.0000 |
| pair_exposed | suffix_only | MR | ba | 0.0000 |
| pair_exposed | suffix_only | NT | ab | 0.0000 |
| pair_exposed | suffix_only | NT | ba | 0.0000 |
| pair_exposed | suffix_only | RM | ab | 0.0000 |
| pair_exposed | suffix_only | RM | ba | 0.0000 |
| pair_exposed | suffix_only | TN | ab | 0.0000 |
| pair_exposed | suffix_only | TN | ba | 0.0000 |
| single_only | article_suffix | MR | ab | 1.0000 |
| single_only | article_suffix | MR | ba | 1.0000 |
| single_only | article_suffix | NT | ab | 1.0000 |
| single_only | article_suffix | NT | ba | 1.0000 |
| single_only | article_suffix | RM | ab | 1.0000 |
| single_only | article_suffix | RM | ba | 1.0000 |
| single_only | article_suffix | TN | ab | 1.0000 |
| single_only | article_suffix | TN | ba | 1.0000 |
| single_only | canonical | MR | ab | 1.0000 |
| single_only | canonical | MR | ba | 1.0000 |
| single_only | canonical | NT | ab | 1.0000 |
| single_only | canonical | NT | ba | 1.0000 |
| single_only | canonical | RM | ab | 1.0000 |
| single_only | canonical | RM | ba | 1.0000 |
| single_only | canonical | TN | ab | 1.0000 |
| single_only | canonical | TN | ba | 1.0000 |
| single_only | suffix_only | MR | ab | 0.0000 |
| single_only | suffix_only | MR | ba | 0.0000 |
| single_only | suffix_only | NT | ab | 0.0000 |
| single_only | suffix_only | NT | ba | 0.0000 |
| single_only | suffix_only | RM | ab | 0.0000 |
| single_only | suffix_only | RM | ba | 0.0000 |
| single_only | suffix_only | TN | ab | 0.0000 |
| single_only | suffix_only | TN | ba | 0.0000 |

Interpretation: canonical and article_suffix are seen training templates; suffix_only is the held-out surface combination. These controls separate failure of the operation itself from failure to recombine surface structure.
