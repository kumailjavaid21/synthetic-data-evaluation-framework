"""
Base-paper evaluation metrics for synthetic tabular data.
Includes divergence metrics, TSTR utility, and privacy risks.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

# Defaults for quasi-identifiers per dataset (column indices)
QI_DEFAULT = {
    "A": [0, 1, 2],
    "B": [0, 1, 2],
    "C": [0, 1, 2],
}

LABEL_DEFAULT_IDX = -1  # assume last column is label if not provided


def load_data(dataset: str, run_dir: Path):
    real_train = np.load(Path("data") / f"{dataset}_Xtr.npy")
    real_test_path = Path("data") / f"{dataset}_Xte.npy"
    if not real_test_path.exists():
        raise FileNotFoundError(f"Missing real test/holdout: {real_test_path}")
    real_test = np.load(real_test_path)

    synth_path = run_dir / "synth.npy"
    if synth_path.exists():
        synth = np.load(synth_path)
    else:
        synth_csv = run_dir / "synth.csv"
        if not synth_csv.exists():
            raise FileNotFoundError(f"No synth.npy/csv found in {run_dir}")
        synth = pd.read_csv(synth_csv).to_numpy()

    synth = np.asarray(synth, dtype=float)
    real_train = np.asarray(real_train, dtype=float)
    real_test = np.asarray(real_test, dtype=float)
    synth = np.nan_to_num(synth)
    real_train = np.nan_to_num(real_train)
    real_test = np.nan_to_num(real_test)
    return real_train, real_test, synth


def load_privacy_metadata(run_dir: Path):
    """Load optional privacy metadata from privacy.json and stats.npz."""
    meta = {}
    privacy_path = run_dir / "privacy.json"
    if privacy_path.exists():
        try:
            with open(privacy_path) as f:
                pj = json.load(f)
            for k, v in pj.items():
                if np.isscalar(v) or isinstance(v, (str, bool)):
                    meta[k] = v
        except Exception as e:
            print(f"[WARN] Failed to read privacy.json at {privacy_path}: {e}")
    stats_path = run_dir / "stats.npz"
    if stats_path.exists():
        try:
            stats = np.load(stats_path)
            for key in [
                "epsilon_target",
                "epsilon_train_target",
                "epsilon_repair",
                "epsilon_achieved_train",
                "epsilon_achieved_total",
                "delta",
                "q",
                "batch_size",
                "n_samples",
                "epochs",
                "seed",
            ]:
                if key in stats:
                    meta[key] = float(stats[key])
        except Exception as e:
            print(f"[WARN] Failed to read stats.npz at {stats_path}: {e}")
    return meta


def _hist_from_bins(data, bins):
    hist, _ = np.histogram(data, bins=bins)
    return hist


def _resolve_label_idx(width, label_idx):
    idx = label_idx if label_idx >= 0 else width + label_idx
    if idx < 0 or idx >= width:
        raise ValueError(f"Label index {label_idx} is out of bounds for width {width}")
    return idx


def _drop_label(arr, label_idx):
    idx = _resolve_label_idx(arr.shape[1], label_idx)
    return np.delete(arr, idx, axis=1)


def divergence_metrics(real, synth, num_bins=20, eps=1e-6):
    cols = real.shape[1]
    per_feat = []
    for j in range(cols):
        r_col = real[:, j]
        s_col = synth[:, j]
        unique = np.unique(r_col)
        if len(unique) <= 20:
            # categorical
            vals = unique
            r_counts = np.array([(r_col == v).sum() for v in vals], dtype=float)
            s_counts = np.array([(s_col == v).sum() for v in vals], dtype=float)
        else:
            mn, mx = np.min(r_col), np.max(r_col)
            if mn == mx:
                per_feat.append({"feature": j, "kld": 0.0, "chi2": 0.0, "jsd_sqrt": 0.0, "w1": 0.0})
                continue
            bins = np.linspace(mn, mx, num_bins + 1)
            r_counts = _hist_from_bins(r_col, bins)
            s_counts = _hist_from_bins(s_col, bins)
            vals = None  # not used
        r_probs = (r_counts + eps) / (r_counts.sum() + eps * len(r_counts))
        s_probs = (s_counts + eps) / (s_counts.sum() + eps * len(s_counts))
        kld = float(entropy(r_probs, s_probs))
        chi2 = float(((r_probs - s_probs) ** 2 / (r_probs + eps)).sum())
        jsd_sqrt = float(jensenshannon(r_probs, s_probs))
        w1 = float(wasserstein_distance(np.arange(len(r_probs)), np.arange(len(r_probs)), r_probs, s_probs))
        per_feat.append({"feature": j, "kld": kld, "chi2": chi2, "jsd_sqrt": jsd_sqrt, "w1": w1})
    # aggregate
    agg = {
        "kld_mean": float(np.mean([d["kld"] for d in per_feat])),
        "chi2_mean": float(np.mean([d["chi2"] for d in per_feat])),
        "jsd_sqrt_mean": float(np.mean([d["jsd_sqrt"] for d in per_feat])),
        "w1_mean": float(np.mean([d["w1"] for d in per_feat])),
        "per_feature": per_feat,
    }
    return agg


def tstr_auc(real_train, real_test, synth, label_idx=LABEL_DEFAULT_IDX):
    if label_idx is None:
        raise ValueError("Label index must be provided")

    all_real = np.vstack([real_train, real_test]) if len(real_test) else real_train
    idx = _resolve_label_idx(all_real.shape[1], label_idx)

    def split_xy(arr):
        X = np.delete(arr, idx, axis=1)
        y = arr[:, idx]
        return X, y

    X_real, y_real = split_xy(all_real)
    X_synth, y_synth = split_xy(synth)

    uniq_real = np.unique(y_real)
    uniq_synth = np.unique(y_synth)
    is_binary_real = len(uniq_real) == 2 and np.allclose(uniq_real, np.round(uniq_real))
    is_binary_synth = len(uniq_synth) == 2 and np.allclose(uniq_synth, np.round(uniq_synth))
    if not (is_binary_real and is_binary_synth):
        # If labels are continuous or not binary, skip TSTR and return NaNs
        return {"auc_real": np.nan, "auc_tstr": np.nan, "utility_loss": np.nan}

    cat_cols = [i for i in range(X_real.shape[1]) if len(np.unique(X_real[:, i])) <= 20]
    num_cols = [i for i in range(X_real.shape[1]) if i not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    preproc = ColumnTransformer(transformers)
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([("pre", preproc), ("clf", clf)])

    def _safe_stratify(y):
        uniq, counts = np.unique(y, return_counts=True)
        return y if (len(uniq) > 1 and counts.min() >= 2) else None

    strat_real = _safe_stratify(y_real)
    Xr_train, Xr_holdout, yr_train, yr_holdout = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=strat_real
    )
    if len(np.unique(yr_holdout)) < 2:
        return {"auc_real": np.nan, "auc_tstr": np.nan, "utility_loss": np.nan}
    pipe.fit(Xr_train, yr_train)
    auc_real = float(roc_auc_score(yr_holdout, pipe.predict_proba(Xr_holdout)[:, 1]))

    strat_synth = _safe_stratify(y_synth)
    Xs_train, _, ys_train, _ = train_test_split(
        X_synth, y_synth, test_size=0.3, random_state=42, stratify=strat_synth
    )
    pipe.fit(Xs_train, ys_train)
    auc_tstr = float(roc_auc_score(yr_holdout, pipe.predict_proba(Xr_holdout)[:, 1]))

    util_loss = auc_real - auc_tstr
    return {"auc_real": auc_real, "auc_tstr": auc_tstr, "utility_loss": util_loss}


def k_anonymity_metrics(X_synth, qis, ks=(2, 5, 10)):
    if not qis:
        return {f"frac_k{k}": None for k in ks}
    qi_view = X_synth[:, qis]
    df_qi = pd.DataFrame(qi_view)
    counts = df_qi.value_counts().to_dict()
    total = len(df_qi)
    res = {}
    for k in ks:
        frac = sum(v < k for v in counts.values()) / total if total else None
        res[f"frac_k{k}"] = frac
    return res


def delta_presence(X_real, X_synth, qis):
    if not qis:
        return {"delta_max": None, "delta_mean": None}
    df_real = pd.DataFrame(X_real[:, qis])
    df_synth = pd.DataFrame(X_synth[:, qis])
    real_counts = df_real.value_counts().to_dict()
    synth_counts = df_synth.value_counts().to_dict()
    deltas = []
    for cls, rc in real_counts.items():
        sc = synth_counts.get(cls, 0)
        delta = sc / rc
        deltas.append(delta)
    if not deltas:
        return {"delta_max": None, "delta_mean": None}
    return {"delta_max": float(np.max(deltas)), "delta_mean": float(np.mean(deltas))}


def identifiability_score(X_real, X_synth, threshold=0.05):
    if len(X_real) == 0 or len(X_synth) == 0:
        return {"identifiability": None, "nn_min": None, "nn_median": None}
    # normalize
    mu = X_real.mean(axis=0, keepdims=True)
    std = X_real.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xr = (X_real - mu) / std
    Xs = (X_synth - mu) / std
    neigh = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neigh.fit(Xr)
    dists, _ = neigh.kneighbors(Xs, return_distance=True)
    d = dists.squeeze()
    ident = float(np.mean(d <= threshold))
    return {"identifiability": ident, "nn_min": float(np.min(d)), "nn_median": float(np.median(d))}


def eval_run(dataset, run_dir: Path, qis=None, label_idx=LABEL_DEFAULT_IDX, ident_threshold=0.05):
    real_train, real_test, synth = load_data(dataset, run_dir)
    label_idx_resolved = _resolve_label_idx(real_train.shape[1], label_idx)
    real_no_label = _drop_label(real_train, label_idx_resolved)
    synth_no_label = _drop_label(synth, label_idx_resolved)

    div = divergence_metrics(real_no_label, synth_no_label)
    tstr = tstr_auc(real_train, real_test, synth, label_idx=label_idx_resolved)
    qi_list = qis if qis is not None else QI_DEFAULT.get(dataset, [])
    k_anon = k_anonymity_metrics(synth_no_label, qi_list)
    delta = delta_presence(real_no_label, synth_no_label, qi_list)
    ident = identifiability_score(real_no_label, synth_no_label, threshold=ident_threshold)
    meta = load_privacy_metadata(run_dir)
    res = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        **div,
        "auc_real": tstr["auc_real"],
        "auc_tstr": tstr["auc_tstr"],
        "utility_loss": tstr["utility_loss"],
        **k_anon,
        **delta,
        **ident,
        **meta,
    }
    return res, div["per_feature"]
