from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "experiments" / "glt_steer_confirmatory_fixed_params_20260825_results" / "csv" / "glt_steer_confirmatory_raw.csv"
OUT_DIR = ROOT / "results" / "experiments" / "glt_steer_source_cluster_bootstrap_20261009_results"
CSV_DIR = OUT_DIR / "csv"
OUT_CSV = CSV_DIR / "glt_steer_source_cluster_bootstrap.csv"
SUMMARY = OUT_DIR / "SUMMARY.md"
BOOTSTRAPS = 5000
SEED = 20261009


def bootstrap_cluster_rates(frame: pd.DataFrame, metrics: list[str], rng: np.random.Generator) -> dict[str, float]:
    source_means = frame.groupby("source_id", as_index=False)[metrics].mean(numeric_only=True)
    values = source_means[metrics].to_numpy(dtype=float)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    sampled = values[draws].mean(axis=1)
    result: dict[str, float] = {}
    for idx, metric in enumerate(metrics):
        point = float(values[:, idx].mean())
        lo, hi = np.quantile(sampled[:, idx], [0.025, 0.975])
        result[f"{metric}_estimate"] = point
        result[f"{metric}_ci95_low"] = float(lo)
        result[f"{metric}_ci95_high"] = float(hi)
    return result


def main() -> None:
    raw = pd.read_csv(INPUT)
    metrics = ["target_marker_hit", "target_and_preserved"]
    required = {"source_id", "model", "target_class", "control", *metrics}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    group_cols = ["model", "target_class", "control"]
    for keys, frame in raw.groupby(group_cols, dropna=False):
        model, target_class, control = keys
        result = bootstrap_cluster_rates(frame, metrics, rng)
        rows.append(
            {
                "model": model,
                "target_class": target_class,
                "control": control,
                "source_clusters": int(frame["source_id"].nunique()),
                "rows": int(len(frame)),
                "prompt_layer_replicates_per_source": int(len(frame) / frame["source_id"].nunique()),
                "bootstrap_replicates": BOOTSTRAPS,
                **result,
            }
        )

    table = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "bootstrap_unit": "source_id",
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "groups": int(len(table)),
    }
    (OUT_DIR / "run_status.json").write_text(json.dumps({"status": "finished", **summary}, indent=2), encoding="utf-8")

    lines = [
        "# GLT-STEER Source-Cluster Bootstrap",
        "",
        "This derived audit recomputes uncertainty for the fixed-parameter confirmatory GLT-STEER run with the independent sentence source as the resampling unit. Prompt styles and layers remain repeated measurements within each source cluster; they are not treated as independent sources.",
        "",
        f"- Input: `{INPUT.relative_to(ROOT)}`",
        f"- Bootstrap unit: `source_id`",
        f"- Replicates: `{BOOTSTRAPS}`",
        f"- Seed: `{SEED}`",
        "- Intervals: percentile 95% cluster-bootstrap intervals.",
        "",
        "The intervals are descriptive and do not repair the synthetic-template limitation. They provide a more conservative uncertainty check than row-level Wilson intervals for the confirmatory source set.",
        "",
        "## Selected results",
        "",
        "| model | target | control | sources | marker rate (95% cluster CI) | marker+preserved (95% cluster CI) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        if row.control not in {"none", "target", "wrong_marker", "random_norm", "negative_target"}:
            continue
        lines.append(
            f"| `{row.model}` | `{row.target_class}` | `{row.control}` | {row.source_clusters} | "
            f"{row.target_marker_hit_estimate:.4f} [{row.target_marker_hit_ci95_low:.4f}, {row.target_marker_hit_ci95_high:.4f}] | "
            f"{row.target_and_preserved_estimate:.4f} [{row.target_and_preserved_ci95_low:.4f}, {row.target_and_preserved_ci95_high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The target/control separation remains a source-level descriptive result, but the uncertainty is wider than a row-level interval would suggest because each source contributes multiple prompt and layer rows. This audit does not create new model evidence; it reports the dependence structure explicitly.",
            "",
            "The current GLT-STEER claim remains bounded to final-marker activation steering. It does not establish semantic rewriting or a general linguistic operator.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
