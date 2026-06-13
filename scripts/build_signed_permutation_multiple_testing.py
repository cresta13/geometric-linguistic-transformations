from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = EXP / "lie_algebraic_identities_results" / "csv"


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        prev = val
        adj[order[i]] = val
    return np.clip(adj, 0, 1)


def bootstrap_p_less_than_one(values, n_boot=20000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    p = (np.sum(means >= 1.0) + 1) / (n_boot + 1)
    return float(p)


def main():
    files = [
        EXP / "lie_algebraic_identities_results" / "csv" / "jacobi_raw_all_models.csv",
        EXP / "lie_algebraic_identities_decoder_results" / "csv" / "jacobi_raw_all_models.csv",
    ]
    raw = pd.concat([pd.read_csv(path) for path in files if path.exists()], ignore_index=True)

    rows = []
    for (model, triple), group in raw.groupby(["model", "triple"]):
        ratios = group["jacobi_to_null_mean_ratio"].to_numpy()
        mean_ratio = float(ratios.mean())
        p_less = bootstrap_p_less_than_one(ratios)
        rows.append({
            "model": model,
            "triple": triple,
            "n": int(len(group)),
            "mean_ratio_to_null": mean_ratio,
            "bootstrap_p_ratio_less_than_1": p_less,
            "direction": "below_null" if mean_ratio < 1 else "above_null",
        })

    result = pd.DataFrame(rows).sort_values(["bootstrap_p_ratio_less_than_1", "model", "triple"])
    result["bonferroni_p"] = np.minimum(result["bootstrap_p_ratio_less_than_1"] * len(result), 1.0)
    result["bh_fdr_p"] = benjamini_hochberg(result["bootstrap_p_ratio_less_than_1"].to_numpy())
    result["passes_bonferroni_0_05"] = result["bonferroni_p"] < 0.05
    result["passes_bh_fdr_0_05"] = result["bh_fdr_p"] < 0.05

    out_path = OUT / "signed_permutation_multiple_testing.csv"
    result.to_csv(out_path, index=False)
    print(out_path)
    print(result)


if __name__ == "__main__":
    main()
