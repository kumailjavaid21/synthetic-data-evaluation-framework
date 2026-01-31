import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utility_tstr_and_corr import resolve_label_col, get_or_create_split, SPLIT_SEED


def normalize_path(p: str) -> str:
    s = str(p).strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]:/", "", s)
    s = re.sub(r"^\./", "", s)
    return s


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


def _min_nn_distances(train_arr: np.ndarray, synth_arr: np.ndarray, chunk: int = 256) -> np.ndarray:
    n = synth_arr.shape[0]
    mins = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        x = synth_arr[start:end]
        dists = np.sum((x[:, None, :] - train_arr[None, :, :]) ** 2, axis=2)
        mins[start:end] = np.sqrt(np.min(dists, axis=1))
    return mins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="outputs/master_results.csv")
    parser.add_argument("--out_csv", type=str, default="outputs/knn_mia_memorization_audit_C_dpcr.csv")
    args = parser.parse_args()

    master_path = Path(args.master_csv)
    if not master_path.exists():
        raise FileNotFoundError(master_path)

    df = pd.read_csv(master_path)
    if "dataset" not in df.columns or "method" not in df.columns:
        raise ValueError("master_results.csv missing dataset/method columns")

    df["knn_mia_auc"] = pd.to_numeric(df.get("knn_mia_auc"), errors="coerce")
    df["knn_mia_auc_raw"] = pd.to_numeric(df.get("knn_mia_auc_raw"), errors="coerce")

    mask = df["dataset"].astype(str).str.upper().eq("C") & df["method"].astype(str).str.contains(
        "DP-Diffusion+DP-CR", case=False, na=False
    )
    df = df[mask]
    df = df[(df["knn_mia_auc"] >= 0.999) | (df["knn_mia_auc_raw"] >= 0.999)]

    rows = []
    for _, row in df.iterrows():
        syn_path = row.get("syn_path")
        if not isinstance(syn_path, str) or not syn_path:
            continue
        syn_file = Path(syn_path)
        if not syn_file.exists():
            print(f"[WARN] Missing synth file: {syn_file}")
            continue

        try:
            synth = pd.read_csv(syn_file)
            label_col, _, _ = resolve_label_col("C", synth)
            if label_col is None:
                raise ValueError("label_col_not_found")
            feature_cols = [c for c in synth.columns if c != label_col]

            train_real, _ = get_or_create_split(
                "C", feature_cols, label_col, split_seed=SPLIT_SEED
            )
            # Align to synth columns (full row match includes label)
            train_aligned = train_real.reindex(columns=synth.columns)

            # Exact match fraction
            synth_hash = _hash_rows(synth)
            train_hash = set(_hash_rows(train_aligned))
            exact_match_fraction = float(np.mean([h in train_hash for h in synth_hash]))

            # Duplicate fraction in synth
            n_rows = len(synth)
            n_unique = int(pd.DataFrame({"h": synth_hash}).nunique()["h"])
            duplicate_fraction = 1.0 - (n_unique / n_rows) if n_rows else np.nan

            # Min NN distances (numeric only)
            synth_num = _numeric_only(synth.drop(columns=[label_col], errors="ignore"))
            train_num = _numeric_only(train_real.drop(columns=[label_col], errors="ignore"))
            nn_min = nn_p01 = nn_median = np.nan
            note = ""
            if synth_num.shape[1] == 0 or train_num.shape[1] == 0:
                note = "no_numeric_cols"
            else:
                # Standardize using train stats
                mean = train_num.mean(axis=0)
                std = train_num.std(axis=0).replace(0, 1.0)
                train_arr = ((train_num - mean) / std).to_numpy(dtype=np.float32)
                synth_arr = ((synth_num - mean) / std).to_numpy(dtype=np.float32)
                mins = _min_nn_distances(train_arr, synth_arr)
                nn_min = float(np.min(mins))
                nn_p01 = float(np.quantile(mins, 0.01))
                nn_median = float(np.median(mins))

            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "epsilon": row.get("epsilon"),
                    "seed": row.get("seed"),
                    "run_dir": row.get("run_dir"),
                    "syn_path": syn_path,
                    "repaired_flag": row.get("repaired_flag"),
                    "synth_kind": row.get("synth_kind"),
                    "knn_mia_auc": row.get("knn_mia_auc"),
                    "knn_mia_auc_raw": row.get("knn_mia_auc_raw"),
                    "exact_match_fraction": exact_match_fraction,
                    "duplicate_fraction": duplicate_fraction,
                    "nn_min": nn_min,
                    "nn_p01": nn_p01,
                    "nn_median": nn_median,
                    "n_synth": len(synth),
                    "n_train": len(train_real),
                    "note": note,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "epsilon": row.get("epsilon"),
                    "seed": row.get("seed"),
                    "run_dir": row.get("run_dir"),
                    "syn_path": syn_path,
                    "repaired_flag": row.get("repaired_flag"),
                    "synth_kind": row.get("synth_kind"),
                    "knn_mia_auc": row.get("knn_mia_auc"),
                    "knn_mia_auc_raw": row.get("knn_mia_auc_raw"),
                    "exact_match_fraction": np.nan,
                    "duplicate_fraction": np.nan,
                    "nn_min": np.nan,
                    "nn_p01": np.nan,
                    "nn_median": np.nan,
                    "n_synth": np.nan,
                    "n_train": np.nan,
                    "note": f"error:{type(e).__name__}",
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")

    print(f"[STAT] audited_rows={len(out_df)}")
    if not out_df.empty:
        sort_cols = ["exact_match_fraction", "nn_p01"]
        for c in sort_cols:
            if c in out_df.columns:
                out_df[c] = pd.to_numeric(out_df[c], errors="coerce")
        worst = out_df.sort_values(by=sort_cols, ascending=[False, True])
        cols = [
            c
            for c in [
                "dataset",
                "method",
                "epsilon",
                "seed",
                "exact_match_fraction",
                "duplicate_fraction",
                "nn_p01",
                "nn_median",
                "syn_path",
            ]
            if c in worst.columns
        ]
        print("[TOP] Worst runs by exact_match_fraction then nn_p01:")
        print(worst[cols].head(10).to_string(index=False))
    else:
        print("[TOP] No rows matched the filter for dataset C DP-CR with knn_mia_auc>=0.999.")


if __name__ == "__main__":
    main()
