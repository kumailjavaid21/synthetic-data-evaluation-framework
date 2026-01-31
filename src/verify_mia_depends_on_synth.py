import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_attacks import knn_distance_attack, save_attack_json


def infer_dataset(run_dir: Path) -> Optional[str]:
    parts = [p.upper() for p in run_dir.parts]
    for ds in ["A", "B", "C", "D"]:
        if ds in parts:
            return ds
    return None


def load_label_col(run_dir: Path, df: pd.DataFrame) -> Optional[str]:
    info_path = run_dir / "label_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            for key in ["target_col", "label_col", "target"]:
                val = info.get(key)
                if isinstance(val, str) and val in df.columns:
                    return val
        except Exception:
            pass
    for cand in ["target", "label", "y"]:
        if cand in df.columns:
            return cand
    return None


def load_members_nonmembers(run_dir: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    idx_members = run_dir / "mia_members_idx.npy"
    idx_nonmembers = run_dir / "mia_nonmembers_idx.npy"
    if not idx_members.exists() or not idx_nonmembers.exists():
        raise FileNotFoundError("Missing mia_members_idx.npy or mia_nonmembers_idx.npy")
    members_idx = np.load(idx_members)
    nonmembers_idx = np.load(idx_nonmembers)
    base = Path("data") / f"{dataset}_Xtr.npy"
    if not base.exists():
        raise FileNotFoundError(f"Real training data not found: {base}")
    X = np.load(base)
    return X[members_idx], X[nonmembers_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify MIA depends on synth.csv by corruption test.")
    ap.add_argument("run_dir", type=Path, help="Run directory containing synth.csv and mia_attack.json")
    args = ap.parse_args()

    run_dir = args.run_dir
    mia_path = run_dir / "mia_attack.json"
    if not mia_path.exists():
        raise FileNotFoundError(mia_path)

    dataset = infer_dataset(run_dir)
    if dataset is None:
        raise ValueError(f"Could not infer dataset from run_dir: {run_dir}")

    orig_mia = json.loads(mia_path.read_text())
    orig_adv = float(orig_mia.get("mia_advantage_abs", np.nan))
    orig_auc = float(orig_mia.get("mia_auc_star", np.nan))

    synth_csv = run_dir / "synth.csv"
    if not synth_csv.exists():
        raise FileNotFoundError(synth_csv)
    synth_backup = run_dir / "synth_backup.csv"
    synth_npy = run_dir / "synth.npy"
    synth_npy_backup = run_dir / "synth_backup.npy"

    df = pd.read_csv(synth_csv)
    label_col = load_label_col(run_dir, df)
    feature_cols = [c for c in df.columns if c != label_col] if label_col else list(df.columns)

    try:
        shutil.copy2(synth_csv, synth_backup)
        if synth_npy.exists():
            shutil.copy2(synth_npy, synth_npy_backup)

        # Corrupt features: zero out feature columns while keeping label column.
        df_corrupt = df.copy()
        if feature_cols:
            df_corrupt[feature_cols] = 0
        df_corrupt.to_csv(synth_csv, index=False)
        if synth_npy.exists():
            np.save(synth_npy, df_corrupt.to_numpy())

        members, nonmembers = load_members_nonmembers(run_dir, dataset)
        synth_arr = df_corrupt.to_numpy()
        res = knn_distance_attack(members, nonmembers, synth_arr, metric="l2", k=1)
        res["dataset"] = dataset
        res["run_dir"] = str(run_dir)
        res["attack"] = "knn"
        save_attack_json(res, run_dir / "mia_attack_test.json")

        test_adv = float(res.get("mia_advantage_abs", np.nan))
        test_auc = float(res.get("mia_auc_star", np.nan))

        diff = abs(test_adv - orig_adv)
        print(f"Original mia_advantage_abs: {orig_adv:.9f} (mia_auc_star={orig_auc:.9f})")
        print(f"Test mia_advantage_abs:      {test_adv:.9f} (mia_auc_star={test_auc:.9f})")
        print(f"Difference > 1e-6: {diff > 1e-6}")
    finally:
        if synth_backup.exists():
            shutil.move(synth_backup, synth_csv)
        if synth_npy_backup.exists():
            shutil.move(synth_npy_backup, synth_npy)


if __name__ == "__main__":
    main()
