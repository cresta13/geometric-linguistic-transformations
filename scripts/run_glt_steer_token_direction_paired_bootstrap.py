"""Source-paired bootstrap for the corrected GLT-STEER token audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "experiments" / "glt_steer_token_direction_controls_20261010_results" / "csv" / "glt_steer_token_direction_controls_raw.csv"
OUT_DIR = ROOT / "results" / "experiments" / "glt_steer_token_direction_paired_bootstrap_20261010_results"
CSV_DIR = OUT_DIR / "csv"
OUT_CSV = CSV_DIR / "glt_steer_token_direction_paired_bootstrap.csv"
SUMMARY_MD = OUT_DIR / "SUMMARY.md"
STATUS = OUT_DIR / "run_status.json"
SEED = 20261010
N_BOOT = 5000
MARGIN = 0.10


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT)
    pair = raw[raw["control"].isin(["target_delta", "target_token_embedding_normmatched"])].copy()
    if pair["source_id"].nunique() != 40:
        raise AssertionError("expected 40 paired held-out sources")
    if pair.groupby("source_id")["control"].nunique().min() != 2:
        raise AssertionError("each source must have both directions")

    rng = np.random.default_rng(SEED)
    rows = []
    for metric in ["target_marker_hit", "target_and_preserved"]:
        wide = pair.pivot(index="source_id", columns="control", values=metric).sort_index()
        diff = wide["target_delta"].to_numpy(float) - wide["target_token_embedding_normmatched"].to_numpy(float)
        estimates = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            estimates[i] = rng.choice(diff, size=len(diff), replace=True).mean()
        lo, hi = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "sources": len(diff),
                "target_delta_rate": float(wide["target_delta"].mean()),
                "target_token_embedding_normmatched_rate": float(wide["target_token_embedding_normmatched"].mean()),
                "paired_difference": float(diff.mean()),
                "bootstrap_ci_low": float(lo),
                "bootstrap_ci_high": float(hi),
                "equivalence_margin": MARGIN,
                "ci_inside_margin": bool(lo > -MARGIN and hi < MARGIN),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    lines = [
        "# GLT-STEER Token-Direction Paired Bootstrap",
        "",
        "This derived audit compares the corrected learned question delta with the norm-matched `?` token embedding on the same held-out source for every pair.",
        "",
        f"- Input: `{relative(INPUT)}`",
        f"- Sources: `40` paired held-out sources",
        f"- Bootstrap replicates: `{N_BOOT}` source-level resamples",
        f"- Predeclared descriptive equivalence margin: `+/-{MARGIN:.2f}` absolute rate",
        "",
        "| metric | delta rate | token rate | paired difference | 95% source-bootstrap CI | CI inside +/-0.10 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| `{row.metric}` | {row.target_delta_rate:.4f} | {row.target_token_embedding_normmatched_rate:.4f} | "
            f"{row.paired_difference:.4f} | [{row.bootstrap_ci_low:.4f}, {row.bootstrap_ci_high:.4f}] | "
            f"{'yes' if row.ci_inside_margin else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The paired difference is the learned delta rate minus the capacity-matched target-token rate. This is a source-cluster uncertainty audit, not a formal TOST equivalence test. It is used here to prevent treating two close point estimates as independent evidence.",
            "",
            "The corrected audit therefore does not support a claim that the learned delta is better than a norm-matched `?` token direction under this protocol.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATUS.write_text(
        json.dumps(
            {
                "status": "finished",
                "input": relative(INPUT),
                "output_csv": relative(OUT_CSV),
                "summary_md": relative(SUMMARY_MD),
                "sources": 40,
                "bootstrap_replicates": N_BOOT,
                "equivalence_margin": MARGIN,
                "failures": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
