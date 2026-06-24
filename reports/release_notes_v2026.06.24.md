# Release Notes: v2026.06.24

This release freezes the current research artifact after the 2026-06-24 endpoint-residualization audits for the Lie-style / signed-permutation track.

Zenodo concept DOI / previous record: [10.5281/zenodo.20680414](https://doi.org/10.5281/zenodo.20680414)

It should be archived on Zenodo as **Software** / reproducibility package, not as a peer-reviewed publication. The draft papers are included as research notes inside the software artifact.

## Main Addition

This release adds two endpoint-artifact stress tests for the multilingual third-order signed-permutation diagnostic:

- `scripts/run_lie_endpoint_residualization_audit.py`
- `scripts/run_lie_endpoint_subspace_residualization_audit.py`

New result directories:

- `results/experiments/lie_endpoint_residualization_results/`
- `results/experiments/lie_endpoint_subspace_residualization_results/`

## Endpoint Sign Residualization

The first audit removes the direction learned by a cyclic-versus-anticyclic endpoint-sign probe and recomputes the signed-permutation ratio.

Global mean ratios remain nearly unchanged:

| Triple | Raw ratio | Sign-residualized ratio |
|---|---:|---:|
| `NMT` | `0.763315` | `0.763161` |
| `NQM` | `0.546052` | `0.537218` |
| `NQT` | `0.676048` | `0.671525` |
| `QMT` | `0.590154` | `0.590166` |

## Endpoint Subspace Residualization

The second audit removes endpoint-derived probe subspaces before recomputing the signed-permutation ratio:

- endpoint sign / cyclicity
- triple label from single endpoint deltas
- endpoint position among the six third-order endpoints
- all three subspaces jointly

Global mean ratios:

| Triple | Raw | Remove sign | Remove triple label | Remove endpoint position | Remove all |
|---|---:|---:|---:|---:|---:|
| `NMT` | `0.764073` | `0.763916` | `0.763325` | `0.764539` | `0.764100` |
| `NQM` | `0.543409` | `0.534386` | `0.544664` | `0.537042` | `0.538300` |
| `NQT` | `0.676606` | `0.671906` | `0.679131` | `0.675365` | `0.677458` |
| `QMT` | `0.589554` | `0.589631` | `0.589615` | `0.602188` | `0.603670` |

Endpoint probes confirm that endpoint deltas do encode task information:

| Probe | Mean macro F1 | Chance |
|---|---:|---:|
| cyclic versus anticyclic from endpoint delta | `0.522386` | `0.500000` |
| endpoint position from endpoint delta | `0.273599` | `0.166667` |
| triple label from single endpoint delta | `0.755258` | `0.250000` |

## Interpretation

The new audits strengthen but do not complete the Track 2 evidence.

They support the narrow claim that the multilingual signed-permutation signal is not explained by a simple linear endpoint-sign, endpoint-position, or triple-label probe subspace. They do not prove a Lie algebra, a formal Jacobi identity, or endpoint-balanced robustness. Endpoint-derived information remains present and measurable, so endpoint-balanced template generation remains the next required control before any submission-style claim.

## Status

This is a citable research snapshot and reproducibility package. It is not a peer-reviewed article release.
