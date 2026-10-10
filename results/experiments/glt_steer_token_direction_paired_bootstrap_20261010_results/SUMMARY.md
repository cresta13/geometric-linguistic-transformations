# GLT-STEER Token-Direction Paired Bootstrap

This derived audit compares the corrected learned question delta with the norm-matched `?` token embedding on the same held-out source for every pair.

- Input: `results/experiments/glt_steer_token_direction_controls_20261010_results/csv/glt_steer_token_direction_controls_raw.csv`
- Sources: `40` paired held-out sources
- Bootstrap replicates: `5000` source-level resamples
- Predeclared descriptive equivalence margin: `+/-0.10` absolute rate

| metric | delta rate | token rate | paired difference | 95% source-bootstrap CI | CI inside +/-0.10 |
|---|---:|---:|---:|---:|:---:|
| `target_marker_hit` | 0.3000 | 0.3250 | -0.0250 | [-0.2250, 0.1750] | no |
| `target_and_preserved` | 0.0500 | 0.0250 | 0.0250 | [-0.0500, 0.1250] | no |

The paired difference is the learned delta rate minus the capacity-matched target-token rate. This is a source-cluster uncertainty audit, not a formal TOST equivalence test. It is used here to prevent treating two close point estimates as independent evidence.

The corrected audit therefore does not support a claim that the learned delta is better than a norm-matched `?` token direction under this protocol.
