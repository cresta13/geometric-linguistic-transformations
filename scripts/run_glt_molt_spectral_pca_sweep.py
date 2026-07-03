from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_glt_molt_spectral_nulls.py"
OUT_ROOT = Path(os.getenv(
    "GLT_MOLT_PCA_SWEEP_OUT_ROOT",
    "results/experiments/glt_molt_spectral_pca_sweep_9m_160t_a100_300null_g256_results",
))


def parse_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    pca_dims = parse_list("GLT_MOLT_PCA_SWEEP_DIMS", "64,128,256")
    alphas = os.getenv("GLT_MOLT_PCA_SWEEP_ALPHAS", "100")
    nulls = os.getenv("GLT_MOLT_PCA_SWEEP_NULLS", "300")
    givens = os.getenv("GLT_MOLT_PCA_SWEEP_GIVENS", "256")
    templates = os.getenv("GLT_MOLT_PCA_SWEEP_TEMPLATES_PER_LANGUAGE", "160")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUT_ROOT / "pca_sweep_status.json"
    runs = []

    print("GLT-MOLT SPECTRAL PCA SWEEP", flush=True)
    print(json.dumps({
        "out_root": str(OUT_ROOT),
        "pca_dims": pca_dims,
        "alphas": alphas,
        "nulls": nulls,
        "givens": givens,
        "templates_per_language": templates,
    }, indent=2), flush=True)

    for dim in pca_dims:
        out_dir = OUT_ROOT / f"pca_{dim}"
        env = os.environ.copy()
        env["GLT_MOLT_SPECTRAL_OUT_DIR"] = str(out_dir)
        env["GLT_MOLT_SPECTRAL_ALPHAS"] = alphas
        env["GLT_MOLT_SPECTRAL_NULLS"] = nulls
        env["GLT_MOLT_SPECTRAL_GIVENS"] = givens
        env["GLT_MOLT_TEMPLATES_PER_LANGUAGE"] = templates
        env["GLT_MOLT_PCA_DIM"] = dim

        print(f"\n=== PCA DIM {dim} ===", flush=True)
        started = time.ctime()
        code = subprocess.call([sys.executable, str(SCRIPT)], cwd=str(ROOT), env=env)
        run = {
            "pca_dim": int(dim),
            "out_dir": str(out_dir),
            "started_at": started,
            "finished_at": time.ctime(),
            "return_code": code,
        }
        runs.append(run)
        status_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
        if code != 0:
            raise SystemExit(code)

    status_path.write_text(json.dumps({"runs": runs, "finished_at": time.ctime()}, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
