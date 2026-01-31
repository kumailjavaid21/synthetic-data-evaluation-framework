from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df.empty:
        return None
    lower_map = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        for col in df.columns:
            if key in col.lower().replace(" ", ""):
                return col
    return None


def ensure_advantage_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mia_auc_col = find_col(out, ["mia_auc", "mia_attack_auc", "miaAUC"])
    knn_auc_col = find_col(out, ["knn_mia_auc", "knn_auc", "knn_attack_auc", "knnMIAAUC"])

    if mia_auc_col is None:
        out["mia_adv_fix"] = np.nan
        out["mia_auc_invalid"] = False
        out["mia_auc_missing"] = True
    else:
        mia_auc = pd.to_numeric(out[mia_auc_col], errors="coerce")
        mia_invalid = mia_auc.notna() & ((mia_auc < 0) | (mia_auc > 1))
        out["mia_auc_invalid"] = mia_invalid
        out["mia_auc_missing"] = mia_auc.isna()
        out["mia_adv_fix"] = (mia_auc - 0.5).abs()
        out.loc[mia_invalid, "mia_adv_fix"] = np.nan
        if mia_auc.isna().all():
            out["mia_adv_fix"] = np.nan
            out["mia_auc_missing"] = True

    if knn_auc_col is None:
        out["knn_mia_adv_fix"] = np.nan
        out["knn_mia_auc_invalid"] = False
        out["knn_mia_auc_missing"] = True
    else:
        knn_auc = pd.to_numeric(out[knn_auc_col], errors="coerce")
        knn_invalid = knn_auc.notna() & ((knn_auc < 0) | (knn_auc > 1))
        out["knn_mia_auc_invalid"] = knn_invalid
        out["knn_mia_auc_missing"] = knn_auc.isna()
        out["knn_mia_adv_fix"] = (knn_auc - 0.5).abs()
        out.loc[knn_invalid, "knn_mia_adv_fix"] = np.nan
        if knn_auc.isna().all():
            out["knn_mia_adv_fix"] = np.nan
            out["knn_mia_auc_missing"] = True

    return out


def choose_primary_privacy_adv(df: pd.DataFrame) -> pd.Series:
    if "nn_dist_norm_median_log10" in df.columns:
        nn_vals = pd.to_numeric(df["nn_dist_norm_median_log10"], errors="coerce")
        if nn_vals.notna().any():
            df["privacy_adv_primary_higher_better"] = True
            return nn_vals.rename("privacy_adv_primary")
    if "nn_dist_norm_median" in df.columns:
        nn_vals = pd.to_numeric(df["nn_dist_norm_median"], errors="coerce")
        if nn_vals.notna().any():
            df["privacy_adv_primary_higher_better"] = True
            return nn_vals.rename("privacy_adv_primary")
    df["privacy_adv_primary_higher_better"] = False
    if "knn_mia_adv_fix" not in df.columns:
        df = ensure_advantage_columns(df)
    primary = df["knn_mia_adv_fix"].where(df["knn_mia_adv_fix"].notna(), df["mia_adv_fix"])
    return primary.rename("privacy_adv_primary")


def privacy_metric_higher_better(df: pd.DataFrame, metric_name: str | None) -> bool:
    if not metric_name:
        return False
    if metric_name in {"nn_dist_norm_median", "nn_dist_norm_median_log10"}:
        return True
    if metric_name == "privacy_adv_primary":
        flag = df.get("privacy_adv_primary_higher_better")
        if flag is None:
            return False
        try:
            return bool(pd.to_numeric(flag, errors="coerce").fillna(0).astype(int).max())
        except Exception:
            return bool(flag.any())
    return False


def summarize_privacy_by_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        print("[WARN] Empty dataframe for privacy summary.")
        return
    df = ensure_advantage_columns(df)
    for ds, sub in df.groupby("dataset", dropna=False):
        mia_auc = pd.to_numeric(sub["mia_auc"], errors="coerce") if "mia_auc" in sub.columns else pd.Series(dtype=float)
        knn_auc = pd.to_numeric(sub["knn_mia_auc"], errors="coerce") if "knn_mia_auc" in sub.columns else pd.Series(dtype=float)
        print(f"[DATASET] {ds} n={len(sub)}")
        if mia_auc.notna().any():
            print(
                f"  MIA AUC: min={mia_auc.min():.6f} med={mia_auc.median():.6f} max={mia_auc.max():.6f} "
                f"invalid={int(sub.get('mia_auc_invalid', pd.Series(False)).sum())}"
            )
        if knn_auc.notna().any():
            print(
                f"  KNN AUC: min={knn_auc.min():.6f} med={knn_auc.median():.6f} max={knn_auc.max():.6f} "
                f">=0.99={int((knn_auc >= 0.99).sum())} <=0.01={int((knn_auc <= 0.01).sum())}"
            )
        if "mia_adv_fix" in sub.columns:
            mia_adv = pd.to_numeric(sub["mia_adv_fix"], errors="coerce")
            if mia_adv.notna().any():
                print(
                    f"  MIA ADV: min={mia_adv.min():.6f} med={mia_adv.median():.6f} max={mia_adv.max():.6f}"
                )
        if "knn_mia_adv_fix" in sub.columns:
            knn_adv = pd.to_numeric(sub["knn_mia_adv_fix"], errors="coerce")
            if knn_adv.notna().any():
                print(
                    f"  KNN ADV: min={knn_adv.min():.6f} med={knn_adv.median():.6f} max={knn_adv.max():.6f} "
                    f">=0.49={int((knn_adv >= 0.49).sum())}"
                )
