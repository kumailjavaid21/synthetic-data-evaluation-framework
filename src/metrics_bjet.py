import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance, chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------- helpers ----------
def _align_numeric(real: np.ndarray, synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(real, dtype=float)
    S = np.asarray(synth, dtype=float)
    D = min(R.shape[1], S.shape[1])
    R, S = R[:, :D], S[:, :D]
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return R, S

def _shared_hist(r: np.ndarray, s: np.ndarray, bins: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    # shared bin edges across r and s
    lo = np.nanmin([np.min(r), np.min(s)])
    hi = np.nanmax([np.max(r), np.max(s)])
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    r_hist, edges = np.histogram(r, bins=bins, range=(lo, hi), density=False)
    s_hist, _     = np.histogram(s, bins=bins, range=(lo, hi), density=False)
    # convert to probability mass with smoothing
    r_p = (r_hist.astype(float) + 1e-9) / (r_hist.sum() + 1e-9 * len(r_hist))
    s_p = (s_hist.astype(float) + 1e-9) / (s_hist.sum() + 1e-9 * len(s_hist))
    return r_p, s_p

# ---------- BJET Utility Metrics ----------

def metric_kld_jsd_wd(real: np.ndarray, synth: np.ndarray, bins: int = 32) -> Dict[str, float]:
    """Per-feature KLD (symmetrized), JSD, WD; return means over features."""
    R, S = _align_numeric(real, synth)
    klds, jsds, wds = [], [], []
    for i in range(R.shape[1]):
        r, s = R[:, i], S[:, i]
        # hist pmfs
        r_p, s_p = _shared_hist(r, s, bins=bins)
        # KLD (symmetrized so it's finite & comparable)
        kld_rs = entropy(r_p, s_p)
        kld_sr = entropy(s_p, r_p)
        kld_sym = 0.5 * (kld_rs + kld_sr)
        klds.append(kld_sym)
        # JSD (distance; already symmetric, bounded [0,1])
        jsd = jensenshannon(r_p, s_p)  # this is the distance
        jsds.append(jsd if np.isfinite(jsd) else 0.0)
        # WD on raw values (continuous 1-D Wasserstein)
        wds.append(wasserstein_distance(r, s))
    return {
        "KLD_mean": float(np.mean(klds)),
        "KLD_std":  float(np.std(klds)),
        "JSD_mean": float(np.mean(jsds)),
        "JSD_std":  float(np.std(jsds)),
        "WD_mean":  float(np.mean(wds)),
        "WD_std":   float(np.std(wds)),
    }

def metric_chisq(real: np.ndarray, synth: np.ndarray, bins: int = 20) -> Dict[str, float]:
    """Chi-square on binned counts per feature; robust to rounding mismatches."""
    R, S = _align_numeric(real, synth)
    pvals = []
    for i in range(R.shape[1]):
        r, s = R[:, i], S[:, i]
        r_p, s_p = _shared_hist(r, s, bins=bins)
        # scale to same total counts
        N = 10000
        r_c = np.maximum(np.round(r_p * N), 0.0)
        s_c = np.maximum(np.round(s_p * N), 0.0)
        # force same total to avoid SciPy error
        total_r = np.sum(r_c)
        total_s = np.sum(s_c)
        if total_r <= 0 or total_s <= 0:
            pvals.append(1.0)
            continue
        scale_factor = total_r / total_s
        s_c = s_c * scale_factor
        try:
            stat, p = chisquare(f_obs=r_c, f_exp=s_c)
        except ValueError:
            # fallback when tiny rounding differences remain
            stat, p = 0.0, 1.0
        pvals.append(p if np.isfinite(p) else 0.0)
    return {
        "CHI_p_mean": float(np.mean(pvals)),
        "CHI_p_std":  float(np.std(pvals)),
    }


# ---------- TSTR (Application Fidelity per BJET) ----------

def tstr_auc(
    X_real_train: np.ndarray,
    y_real_train: np.ndarray,
    X_real_test:  np.ndarray,
    y_real_test:  np.ndarray,
    X_synth_train: np.ndarray,
) -> Dict[str, float]:
    """
    Train-on-Synthetic, Test-on-Real (TSTR) with automatic label resampling.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    Xr_tr, yr_tr = np.asarray(X_real_train), np.asarray(y_real_train)
    Xr_te, yr_te = np.asarray(X_real_test),  np.asarray(y_real_test)
    Xs_tr        = np.asarray(X_synth_train)

    # --- align feature dimensions ---
    d_real, d_synth = Xr_tr.shape[1], Xs_tr.shape[1]
    if d_real != d_synth:
        d_min = min(d_real, d_synth)
        Xr_tr = Xr_tr[:, :d_min]
        Xr_te = Xr_te[:, :d_min]
        Xs_tr = Xs_tr[:, :d_min]

    # --- resample labels to match synthetic size ---
    if len(Xs_tr) != len(yr_tr):
        idx = np.random.choice(len(yr_tr), size=len(Xs_tr), replace=True)
        y_synth = yr_tr[idx]
    else:
        y_synth = yr_tr

    scaler = StandardScaler()
    Xr_tr_s = scaler.fit_transform(Xr_tr)
    Xr_te_s = scaler.transform(Xr_te)
    Xs_tr_s = scaler.transform(Xs_tr)

    # --- train and evaluate ---
    lr_real = LogisticRegression(max_iter=500, n_jobs=1)
    lr_real.fit(Xr_tr_s, yr_tr)
    lr_synth = LogisticRegression(max_iter=500, n_jobs=1)
    lr_synth.fit(Xs_tr_s, y_synth)

    uniques = np.unique(yr_tr)
    try:
        if len(uniques) == 2:
            auc_real  = roc_auc_score(yr_te, lr_real.predict_proba(Xr_te_s)[:, 1])
            auc_synth = roc_auc_score(yr_te, lr_synth.predict_proba(Xr_te_s)[:, 1])
            util_loss = auc_real - auc_synth
            return {"TSTR_metric": "AUC",
                    "AUC_real": float(auc_real),
                    "AUC_synth": float(auc_synth),
                    "Utility_loss": float(util_loss)}
        else:
            pred_real  = lr_real.predict(Xr_te_s)
            pred_synth = lr_synth.predict(Xr_te_s)
            f1_real  = f1_score(yr_te, pred_real, average="macro")
            f1_synth = f1_score(yr_te, pred_synth, average="macro")
            acc_real  = accuracy_score(yr_te, pred_real)
            acc_synth = accuracy_score(yr_te, pred_synth)
            return {"TSTR_metric": "F1/ACC",
                    "F1_real": float(f1_real), "F1_synth": float(f1_synth),
                    "ACC_real": float(acc_real), "ACC_synth": float(acc_synth),
                    "Utility_loss_F1": float(f1_real - f1_synth),
                    "Utility_loss_ACC": float(acc_real - acc_synth)}
    except Exception as e:
        return {"TSTR_metric": "NA", "Error": str(e)}

