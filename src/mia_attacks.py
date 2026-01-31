"""
Membership inference attacks for synthetic data.
Implements a distance-based kNN attack against synthetic samples.
"""

from pathlib import Path
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


def knn_distance_attack(member_X, nonmember_X, synth_X, metric="l2", k=1):
    """
    Distance-to-synthetic kNN attack.
    Returns raw/star AUC, advantage, and distance summaries for members/nonmembers.
    """
    member_X = np.asarray(member_X, dtype=float)
    nonmember_X = np.asarray(nonmember_X, dtype=float)
    synth_X = np.asarray(synth_X, dtype=float)

    if len(member_X) == 0 or len(nonmember_X) == 0 or len(synth_X) == 0:
        raise ValueError("member_X, nonmember_X, and synth_X must be non-empty.")

    # Normalize dimensions defensively
    d = min(member_X.shape[1], nonmember_X.shape[1], synth_X.shape[1])
    member_X = member_X[:, :d]
    nonmember_X = nonmember_X[:, :d]
    synth_X = synth_X[:, :d]

    # Fit kNN on synthetic data
    nn_metric = "euclidean" if metric in ("l2", "euclidean") else ("manhattan" if metric in ("l1", "manhattan") else metric)
    nbrs = NearestNeighbors(n_neighbors=k, metric=nn_metric)
    nbrs.fit(synth_X)

    def dist_stats(arr):
        dists, _ = nbrs.kneighbors(arr, return_distance=True)
        dmin = dists.min(axis=1)
        return {
            "min": float(np.min(dmin)),
            "p05": float(np.quantile(dmin, 0.05)),
            "median": float(np.median(dmin)),
            "mean": float(np.mean(dmin)),
            "p95": float(np.quantile(dmin, 0.95)),
            "all": dmin,
        }

    member_stats = dist_stats(member_X)
    nonmember_stats = dist_stats(nonmember_X)

    scores = np.concatenate([-member_stats["all"], -nonmember_stats["all"]])  # closer -> more likely member
    labels = np.concatenate([np.ones_like(member_stats["all"]), np.zeros_like(nonmember_stats["all"])])
    auc_raw = float(roc_auc_score(labels, scores))
    auc_star = float(max(auc_raw, 1.0 - auc_raw))
    advantage_abs = float(abs(auc_raw - 0.5))

    # Drop raw arrays from stats to keep JSON light
    member_stats = {k: v for k, v in member_stats.items() if k != "all"}
    nonmember_stats = {k: v for k, v in nonmember_stats.items() if k != "all"}

    return {
        "mia_auc_raw": auc_raw,
        "mia_auc_star": auc_star,
        "mia_advantage_abs": advantage_abs,
        "member_dist": member_stats,
        "nonmember_dist": nonmember_stats,
        "metric": nn_metric,
        "k": int(k),
    }


def save_attack_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
