import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances

def _clean_pair(real: np.ndarray, synth: np.ndarray):
    """Align, standardize, and impute to ensure finite numeric arrays."""
    R = np.asarray(real, dtype=float)
    S = np.asarray(synth, dtype=float)

    # truncate to same width if needed (defensive)
    D = min(R.shape[1], S.shape[1])
    R, S = R[:, :D], S[:, :D]

    # drop columns that are all NaN in either set
    keep = ~(~np.isfinite(R)).all(axis=0) & ~(~np.isfinite(S)).all(axis=0)
    R, S = R[:, keep], S[:, keep]
    if R.shape[1] == 0:
        # fallback single dummy column if everything was filtered
        R = np.zeros((R.shape[0], 1))
        S = np.ones((S.shape[0], 1))

    # impute NaN/inf in REAL with column medians
    med = np.nanmedian(np.where(np.isfinite(R), R, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    R = np.where(np.isfinite(R), R, med)

    # impute NaN/inf in SYNTH with real medians (same stats)
    S = np.where(np.isfinite(S), S, med)

    # standardize by REAL stats (avoid zero std)
    mu = R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd < 1e-8] = 1e-8
    Rz = (R - mu) / sd
    Sz = (S - mu) / sd
    return Rz, Sz

def fidelity_metrics(real: np.ndarray, synth: np.ndarray):
    R, S = _clean_pair(real, synth)

    # KS across columns (skip columns with <2 unique values in either set)
    ks_vals = []
    for i in range(R.shape[1]):
        r, s = R[:, i], S[:, i]
        if np.unique(r).size > 1 and np.unique(s).size > 1:
            ks_vals.append(ks_2samp(r, s, alternative="two-sided").statistic)
    ks_vals = np.asarray(ks_vals) if ks_vals else np.array([0.0])

    # correlation difference (NaN-safe)
    def safe_corr(X):
        with np.errstate(invalid='ignore', divide='ignore'):
            C = np.corrcoef(X, rowvar=False)
        C = np.where(np.isfinite(C), C, 0.0)
        return C
    corr_diff = np.mean(np.abs(safe_corr(R) - safe_corr(S)))

    return {
        "KS_mean": float(ks_vals.mean()),
        "KS_std": float(ks_vals.std()),
        "Correlation_diff": float(corr_diff),
    }

def privacy_metrics(real: np.ndarray, synth: np.ndarray, sample_size: int = 1000):
    """Nearest-neighbor privacy metric with normalization and scale stabilization."""
    R = np.asarray(real, dtype=float)
    S = np.asarray(synth, dtype=float)
    D = min(R.shape[1], S.shape[1])
    R, S = R[:, :D], S[:, :D]

    # remove non-finite values
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    # global normalization across both sets
    mean = np.mean(np.vstack([R, S]), axis=0)
    std = np.std(np.vstack([R, S]), axis=0)
    std[std < 1e-6] = 1e-6
    Rn = (R - mean) / std
    Sn = (S - mean) / std

    n = min(sample_size, len(Rn), len(Sn))
    Dmat = pairwise_distances(Rn[:n], Sn[:n], metric="euclidean")
    md = np.min(Dmat, axis=1)

    # remove infinities / outliers
    md = np.nan_to_num(md, nan=0.0, posinf=np.nanmedian(md), neginf=0.0)
    md = np.clip(md, 0, np.percentile(md, 99.5))

    # sanity bounds
    md = np.where(md > 10.0, 10.0, md)

    return {
        "NN_mean": float(np.mean(md)),
        "NN_std": float(np.std(md)),
        "NN_min": float(np.min(md))
    }


def evaluate_pair(real_path, synth_path, name):
    real = np.load(real_path)
    synth = np.load(synth_path)
    f = fidelity_metrics(real, synth)
    p = privacy_metrics(real, synth)
    return {"Dataset": name, **f, **p}
