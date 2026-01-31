import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="outputs_A_IMPROVE/master_results.csv")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    master_path = Path(args.master_csv)
    if not master_path.exists():
        raise FileNotFoundError(master_path)

    df = pd.read_csv(master_path)
    if "syn_path" not in df.columns:
        raise ValueError("master_results.csv missing syn_path column")
    df = df[df["syn_path"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with syn_path found in master_results.csv")

    df = df.drop_duplicates(subset=["syn_path"]).head(args.n_samples)
    if df.empty:
        raise ValueError("No unique syn_path rows available for sampling")

    mia_vals = []
    for _, r in df.iterrows():
        syn_path = Path(str(r["syn_path"]))
        if not syn_path.exists():
            print(f"[WARN] Missing synth file: {syn_path}")
            continue
        md5 = _file_md5(syn_path)
        mia_auc = float(r.get("mia_auc", np.nan))
        knn_mia_auc = float(r.get("knn_mia_auc", np.nan))
        mia_vals.append(mia_auc)
        print(f"run_dir={r.get('run_dir')} syn_path={syn_path} md5={md5}")
        print(f"mia_auc={mia_auc} knn_mia_auc={knn_mia_auc}")

    if len(mia_vals) >= 2:
        mia_vals = np.array(mia_vals, dtype=float)
        if np.nanmax(mia_vals) - np.nanmin(mia_vals) <= 1e-9:
            print("WARNING: MIA AUC constant across different synth outputs. MIA pipeline likely not dependent on synth.")


if __name__ == "__main__":
    main()
