from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_DIR = Path("results/experiments/glt_affect_lexical_specificity_control_results")
OUT_DIR = Path("results/experiments/glt_affect_lexical_contrast_bootstrap_results")
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
for directory in [CSV_DIR, FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_le_zero": float("nan"),
            "n": 0,
        }
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_le_zero": float(np.mean(boot <= 0.0)),
        "n": int(n),
    }


def build_contrasts(opp: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "language", "template_id", "representation"]
    pivot = opp.pivot_table(index=keys, columns="scale", values="row_cosine", aggfunc="mean").reset_index()
    rows = []
    for control in ["random_label", "size", "attention"]:
        if control not in pivot.columns:
            continue
        delta = pivot["affect"] - pivot[control]
        for row, value in zip(pivot.itertuples(index=False), delta):
            rows.append({
                "model": getattr(row, "model"),
                "language": getattr(row, "language"),
                "template_id": int(getattr(row, "template_id")),
                "representation": getattr(row, "representation"),
                "contrast": f"affect_minus_{control}",
                "value": float(value),
            })
    return pd.DataFrame(rows)


def summarize(contrasts: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (representation, contrast), sub in contrasts.groupby(["representation", "contrast"]):
        stats = bootstrap_ci(sub["value"].to_numpy(), n_bootstrap, seed)
        rows.append({
            "representation": representation,
            "contrast": contrast,
            **stats,
        })
    for (model, representation, contrast), sub in contrasts.groupby(["model", "representation", "contrast"]):
        stats = bootstrap_ci(sub["value"].to_numpy(), n_bootstrap, seed)
        rows.append({
            "model": model,
            "representation": representation,
            "contrast": contrast,
            **stats,
        })
    for (language, representation, contrast), sub in contrasts.groupby(["language", "representation", "contrast"]):
        stats = bootstrap_ci(sub["value"].to_numpy(), n_bootstrap, seed)
        rows.append({
            "language": language,
            "representation": representation,
            "contrast": contrast,
            **stats,
        })
    return pd.DataFrame(rows)


def plot_global(summary: pd.DataFrame) -> None:
    global_rows = summary[summary.get("model").isna() & summary.get("language").isna()].copy()
    if global_rows.empty:
        return
    for representation, sub in global_rows.groupby("representation"):
        sub = sub.sort_values("contrast")
        y = np.arange(len(sub))
        mean = sub["mean"].to_numpy()
        low = sub["ci_low"].to_numpy()
        high = sub["ci_high"].to_numpy()
        plt.figure(figsize=(8, 4.8))
        plt.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", capsize=4)
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.yticks(y, sub["contrast"])
        plt.xlabel("Paired mean row-cosine difference")
        plt.title(f"GLT-AFFECT lexical contrast bootstrap ({representation})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"lexical_contrast_bootstrap_{representation}.png", dpi=220, bbox_inches="tight")
        plt.close()


def main() -> None:
    config = {
        "seed": int(os.getenv("GLT_AFFECT_CONTRAST_SEED", "20260626")),
        "n_bootstrap": int(os.getenv("GLT_AFFECT_CONTRAST_BOOTSTRAP", "20000")),
        "input": str(IN_DIR / "csv" / "lexical_opposition_all_models.csv"),
    }
    print("GLT-AFFECT LEXICAL CONTRAST BOOTSTRAP")
    print(json.dumps(config, indent=2), flush=True)
    opp_path = IN_DIR / "csv" / "lexical_opposition_all_models.csv"
    if not opp_path.exists():
        raise FileNotFoundError(opp_path)
    opp = pd.read_csv(opp_path)
    contrasts = build_contrasts(opp)
    contrasts.to_csv(CSV_DIR / "lexical_contrast_cells.csv", index=False)
    summary = summarize(contrasts, config["n_bootstrap"], config["seed"])
    summary.to_csv(CSV_DIR / "lexical_contrast_bootstrap_summary.csv", index=False)
    plot_global(summary)
    status = {
        "finished_at": time.ctime(),
        "failures": [],
        "input_rows": int(len(opp)),
        "contrast_rows": int(len(contrasts)),
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT_DIR / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
