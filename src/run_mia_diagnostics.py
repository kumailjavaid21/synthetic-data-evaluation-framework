"""
run_mia_diagnostics.py

Diagnose MIA leakage for a single run.

Usage (PowerShell):
  python run_mia_diagnostics.py --dataset C --run_dir outputs/C/dp_diffusion/seed42/train1.0 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from audits.privacy_audits import load_real_and_synth


def compute_mia_scores(X_real, X_synth, seed=42):
    n = min(len(X_real), len(X_synth))
    Xr = X_real[:n]
    Xs = X_synth[:n]
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xs))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_s, y_train)
    scores = clf.predict_proba(X_test_s)[:, 1]

    auc_raw = roc_auc_score(y_test, scores)
    auc_star = max(auc_raw, 1 - auc_raw)
    adv_abs = auc_star - 0.5

    # Flip sanity
    auc_flip = roc_auc_score(1 - y_test, scores)
    flip_diff = abs((auc_raw + auc_flip) - 1.0)

    # Score distribution summaries
    members = scores[y_test == 1]
    nonmembers = scores[y_test == 0]
    def summarize(arr):
        return {
            "min": float(np.min(arr)),
            "p05": float(np.percentile(arr, 5)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
        }
    return {
        "auc_raw": float(auc_raw),
        "auc_star": float(auc_star),
        "adv_abs": float(adv_abs),
        "auc_flip": float(auc_flip),
        "flip_diff": float(flip_diff),
        "scores_members": summarize(members),
        "scores_nonmembers": summarize(nonmembers),
    }


def main():
    parser = argparse.ArgumentParser(description="MIA diagnostics for a single run.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    X_real, X_synth = load_real_and_synth(args.dataset, run_dir)

    res = compute_mia_scores(X_real, X_synth, seed=args.seed)

    print(f"Dataset={args.dataset} run_dir={run_dir}")
    print(f"  mia_auc_raw: {res['auc_raw']:.3f}")
    print(f"  mia_auc_star: {res['auc_star']:.3f}")
    print(f"  mia_advantage_abs: {res['adv_abs']:.3f}")
    print(f"  flip_diff: {res['flip_diff']:.6f}")
    print(f"  scores_members: {res['scores_members']}")
    print(f"  scores_nonmembers: {res['scores_nonmembers']}")
    print(f"  sanity_split_auc (features only): {res['auc_raw']:.3f}")
    if res['auc_star'] > 0.8:
        print("CRITICAL: High MIA leakage")
    elif res['auc_star'] > 0.7:
        print("WARNING: Strong MIA leakage")
    if res['flip_diff'] > 0.01:
        print("WARNING: Flip sanity check failed (AUC + AUC_flip != 1)")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": args.dataset,
            "run_dir": str(run_dir),
            "seed": args.seed,
            "mia_auc_raw": res["auc_raw"],
            "mia_auc_star": res["auc_star"],
            "mia_advantage_abs": res["adv_abs"],
            "sanity_split_auc": res["auc_raw"],
            "flip_diff": res["flip_diff"],
            "scores_members_min": res["scores_members"]["min"],
            "scores_members_mean": res["scores_members"]["mean"],
            "scores_nonmembers_min": res["scores_nonmembers"]["min"],
            "scores_nonmembers_mean": res["scores_nonmembers"]["mean"],
        }
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"[OK] Diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
