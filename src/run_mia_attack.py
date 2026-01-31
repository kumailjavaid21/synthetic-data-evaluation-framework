"""
run_mia_attack.py

Run distance-based kNN membership inference attack for a DP synthetic run.

Example:
  python run_mia_attack.py --dataset C --run_dir outputs/C/dp_diffusion/seed42/train1.0 --attack knn --k 1 --metric l2
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

from mia_attacks import knn_distance_attack, save_attack_json


def load_synth(run_dir: Path):
    npy = run_dir / "synth.npy"
    csv = run_dir / "synth.csv"
    if npy.exists():
        return np.load(npy)
    if csv.exists():
        return pd.read_csv(csv).to_numpy()
    raise FileNotFoundError(f"No synth.npy/csv found in {run_dir}")


def load_split(run_dir: Path, dataset: str):
    idx_members = run_dir / "mia_members_idx.npy"
    idx_nonmembers = run_dir / "mia_nonmembers_idx.npy"
    if not idx_members.exists() or not idx_nonmembers.exists():
        raise FileNotFoundError("MIA split files mia_members_idx.npy or mia_nonmembers_idx.npy missing.")
    members_idx = np.load(idx_members)
    nonmembers_idx = np.load(idx_nonmembers)
    base = Path("data") / f"{dataset}_Xtr.npy"
    if not base.exists():
        raise FileNotFoundError(f"Real training data not found: {base}")
    X = np.load(base)
    return X[members_idx], X[nonmembers_idx]


def main():
    ap = argparse.ArgumentParser(description="Run MIA distance-based attack for a synthetic run directory.")
    ap.add_argument("--dataset", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--run_dir", required=True, type=Path, help="Path to run directory containing synth and MIA split.")
    ap.add_argument("--attack", type=str, default="knn", choices=["knn"], help="Attack type (currently only knn).")
    ap.add_argument("--k", type=int, default=1, help="k for kNN attack.")
    ap.add_argument("--metric", type=str, default="l2", help="Distance metric for kNN (l2|l1|manhattan|euclidean).")
    args = ap.parse_args()

    run_dir = args.run_dir
    synth = load_synth(run_dir)
    members, nonmembers = load_split(run_dir, args.dataset)

    if args.attack == "knn":
        res = knn_distance_attack(members, nonmembers, synth, metric=args.metric, k=args.k)
    else:
        raise ValueError(f"Unsupported attack: {args.attack}")

    res["dataset"] = args.dataset
    res["run_dir"] = str(run_dir)
    res["attack"] = args.attack

    out_path = run_dir / "mia_attack.json"
    save_attack_json(res, out_path)
    print(f"[OK] MIA attack results written to {out_path}")


if __name__ == "__main__":
    main()
