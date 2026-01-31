import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def norm_key(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("\\", "/").lower()
    while s.endswith("/"):
        s = s[:-1]
    return s


def read_json_table(path: Path) -> pd.DataFrame:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return pd.DataFrame(data["results"])
        if data and all(isinstance(v, dict) for v in data.values()):
            rows = []
            for k, v in data.items():
                if isinstance(v, dict):
                    r = v.copy()
                    r["run_key"] = k
                    rows.append(r)
            return pd.DataFrame(rows)
        return pd.DataFrame([data])
    return pd.DataFrame()


def score_mia_col(name: str, values: pd.Series) -> int:
    name_l = name.lower()
    if "knn" in name_l:
        return -999
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return -999
    score = 0
    if "mia" in name_l:
        score += 50
    if "attack" in name_l:
        score += 25
    if "auc" in name_l:
        score += 10
    in_range = vals.between(0, 1).mean()
    if in_range >= 0.90:
        score += 10
    else:
        score -= 20
    if vals.notna().sum() >= 20:
        score += 10
    std = vals.std()
    if std >= 1e-6:
        score += 10
    else:
        score -= 30
    if vals.nunique(dropna=True) <= 1:
        score -= 30
    return score


def pick_mia_col(df: pd.DataFrame) -> tuple[str | None, int]:
    best = None
    best_score = -10**9
    for c in df.columns:
        c_l = c.lower()
        if "knn" in c_l:
            continue
        if not any(k in c_l for k in ["mia", "attack", "membership"]):
            continue
        score = score_mia_col(c, df[c])
        if score > best_score:
            best_score = score
            best = c
    return best, best_score


def infer_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    run_dir = None
    syn_path = None
    if "run_dir_key" in df.columns:
        run_dir = df["run_dir_key"]
    elif "run_dir" in df.columns:
        run_dir = df["run_dir"]
    if "syn_path_key" in df.columns:
        syn_path = df["syn_path_key"]
    elif "syn_path" in df.columns:
        syn_path = df["syn_path"]
    elif "synth_path" in df.columns:
        syn_path = df["synth_path"]

    if run_dir is None and syn_path is None:
        path_col = None
        for c in df.columns:
            series = df[c].astype(str)
            if series.str.contains("outputs\\\\|outputs/", regex=True).any():
                path_col = c
                break
        if path_col:
            paths = df[path_col].astype(str)
            syn_path = paths.where(
                paths.str.contains("synth\\.csv|synth_unrepaired\\.csv", case=False, regex=True)
            )
            if syn_path.notna().any():
                run_dir = syn_path.map(lambda s: str(Path(str(s)).parent))
            else:
                # If looks like outputs directory path
                run_dir = paths.where(paths.str.contains("outputs/", case=False, regex=False))

    if run_dir is not None:
        df["run_dir"] = run_dir
        df["run_dir_key"] = df["run_dir"].map(norm_key)
    if syn_path is not None:
        df["syn_path"] = syn_path
        df["syn_path_key"] = df["syn_path"].map(norm_key)
    return df


def dedup_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_nnz"] = df.notna().sum(axis=1)
    df = df.sort_values(by=["_nnz", "_mia_source_score", "_mtime"], ascending=[False, False, False])
    key = "syn_path_key" if "syn_path_key" in df.columns and df["syn_path_key"].notna().any() else "run_dir_key"
    df = df.drop_duplicates(subset=[key], keep="first")
    return df.drop(columns=["_nnz"], errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    files = []
    for ext in ["*.csv", "*.json"]:
        files.extend(outputs_root.rglob(ext))

    total_files = 0
    files_loaded = 0
    files_with_mia = 0
    rows = []
    for p in files:
        name = p.name.lower()
        if name in {"synth.csv", "synth_unrepaired.csv"}:
            continue
        if name in {"mia_results.csv", "mia_results_clean.csv"}:
            continue
        if any(k in name for k in ["tstr", "corr", "summary", "utility"]):
            continue
        path_str = str(p).replace("\\", "/").lower()
        if "/knn_mia/" in path_str or "knn_mia_results" in name or "knn_mia" in name:
            continue
        if (
            "paper_package" in p.parts
            or "final_results_bundle" in p.parts
            or "paper_artifacts" in p.parts
        ):
            continue
        if "master_results" in name or "final_results" in name:
            continue
        if p.stat().st_size > 200 * 1024 * 1024:
            print(f"[WARN] Skip large file: {p}")
            continue
        total_files += 1
        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(p, engine="python")
                except Exception:
                    continue
        else:
            df = read_json_table(p)
        if df.empty:
            continue
        files_loaded += 1
        col, score = pick_mia_col(df)
        if col is None or score < 0:
            continue
        df = infer_keys(df)
        mia_vals = pd.to_numeric(df[col], errors="coerce")
        out = pd.DataFrame(
            {
                "run_dir_key": df.get("run_dir_key"),
                "syn_path_key": df.get("syn_path_key"),
                "run_dir": df.get("run_dir"),
                "syn_path": df.get("syn_path"),
                "mia_auc": mia_vals,
                "_mia_source_file": str(p.relative_to(outputs_root)).replace("\\", "/"),
                "_mia_source_col": col,
                "_mia_source_score": score,
            }
        )
        out["_mtime"] = p.stat().st_mtime
        rows.append(out)
        files_with_mia += 1

    if not rows:
        print("[WARN] No usable MIA column found anywhere under outputs/**.")
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df = dedup_rows(all_df)
    out_path = outputs_root / "mia_results.csv"
    out_clean_path = outputs_root / "mia_results_clean.csv"
    keep_cols = [
        "run_dir_key",
        "syn_path_key",
        "run_dir",
        "syn_path",
        "mia_auc",
        "_mia_source_file",
        "_mia_source_col",
        "_mia_source_score",
    ]
    all_df = all_df[keep_cols]
    all_df.to_csv(out_path, index=False)
    all_df.to_csv(out_clean_path, index=False)

    non_nan = int(all_df["mia_auc"].notna().sum())
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {out_clean_path}")
    print(f"[STAT] total_files_scanned={total_files} files_loaded={files_loaded} files_with_mia_col={files_with_mia}")
    print(f"[STAT] total_rows_out={len(all_df)} mia_auc_non_nan={non_nan}")
    top = all_df[all_df["mia_auc"].notna()].sort_values("mia_auc", ascending=False).head(10)
    cols = ["run_dir_key", "syn_path_key", "mia_auc", "_mia_source_file", "_mia_source_col", "_mia_source_score"]
    if not top.empty:
        print(top[cols].to_string(index=False))
    print(all_df["_mia_source_file"].value_counts().head(10).to_string())
    if all_df["_mia_source_file"].astype(str).str.contains("knn", case=False, na=False).any():
        print("[WARN] Detected MIA source file containing 'knn'.")


if __name__ == "__main__":
    main()
