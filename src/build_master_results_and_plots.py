import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.privacy_metrics_utils import ensure_advantage_columns, choose_primary_privacy_adv
from scripts.plot_style_bjet import (
    apply_bjet_style,
    dataset_bar_style,
    dataset_marker,
    method_color,
    order_methods,
    order_datasets,
)
from scripts.paper_plot_style import (
    set_paper_style,
    paper_figsize,
    savefig_pdf_png,
    style_axes,
    method_color_paper,
)
from src.utility_tstr_and_corr import evaluate_one


PATH_COL_CANDIDATES = ["path", "syn_path", "synth_path", "run_path", "artifact_path", "output_dir", "run_dir"]
CONFIG_COL_CANDIDATES = [
    "epsilon_train",
    "epsilon_repair",
    "epsilon",
    "epsilon_total",
    "eps_train",
    "eps_repair",
    "train_frac",
    "repair_frac",
]
PRIVACY_KEYWORDS = ["mia", "privacy", "leakage", "knn", "attack", "audit"]
MIA_SKIP_FILE_PATTERNS = [
    "privacy_equal_rows",
    "mia_results_incremental",
    "best_baseline_overall_row",
    "compact_summary_with_attack",
    "paper_tables",
    "paper_package",
]

PRIVACY_CANONICAL = [
    "mia_auc",
    "knn_mia_auc",
    "delta_presence",
    "identifiability",
    "leakage_score",
    "leakage_pass",
    "nn_dist_norm_median",
    "duplicate_rate",
    "membership_audit_adv",
]

UTILITY_CANONICAL = [
    "tstr_acc",
    "tstr_balacc",
    "tstr_f1macro",
    "rtr_acc",
    "pearson_mad",
]

DELTA_COLS = ["delta_balacc", "delta_macro_f1", "delta_tstr_acc", "delta_corr"]
KNN_CANDIDATE_COLS = [
    "knn_mia_auc",
    "knn_mia_auc_raw",
    "knn_auc",
    "knn_mia",
    "knn_attack_auc",
    "knn",
    "auc_knn",
    "knnauc",
]


def _normalize_path(p: str) -> str:
    s = str(p).strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]:/", "", s)
    s = re.sub(r"^\./", "", s)
    return s


def normalize_path(p: str) -> str:
    return _normalize_path(p)


def normalize_dir_key(p: str) -> str:
    s = _normalize_path(p)
    for suffix in ["/synth.csv", "/synth_unrepaired.csv", "/synth_base.csv"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def norm_path(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return np.nan
    try:
        return Path(str(p)).resolve().as_posix().lower()
    except Exception:
        return str(p).replace("\\", "/").lower()


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_run_id(df: pd.DataFrame) -> pd.Series:
    path_col = _first_existing_col(df, PATH_COL_CANDIDATES)
    if path_col:
        if path_col in {"run_dir", "output_dir", "run_path", "artifact_path"}:
            return df[path_col].astype(str).map(normalize_dir_key)
        return df[path_col].astype(str).map(normalize_path)

    return _build_run_id_alt(df)


def _build_run_id_alt(df: pd.DataFrame) -> pd.Series:
    parts = []
    for c in ["dataset", "method", "seed"]:
        if c in df.columns:
            parts.append(df[c].astype(str))
    for c in CONFIG_COL_CANDIDATES:
        if c in df.columns:
            parts.append(df[c].astype(str))
    if not parts:
        return pd.Series(["unknown"] * len(df))
    return parts[0].str.cat(parts[1:], sep="|")


def _select_best_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_nnz"] = df.notna().sum(axis=1)
    df = df.sort_values(by="_nnz", ascending=False)
    return df.iloc[[0]].drop(columns=["_nnz"])


def _read_privacy_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(f"[WARN] Privacy file empty: {path}")
                return None
            if df.empty or df.columns.empty:
                print(f"[WARN] Privacy file has no columns: {path}")
                return None
            return df
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                if "results" in data and isinstance(data["results"], list):
                    return pd.DataFrame(data["results"])
                return pd.DataFrame([data])
    except Exception as e:
        print(f"[WARN] Failed to read privacy file {path}: {e}")
    return None


def _read_any_table(path: Path) -> Optional[pd.DataFrame]:
    if path.name.lower() in {"synth.csv", "synth_unrepaired.csv"}:
        return None
    try:
        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return None
            return df if not df.empty else None
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                if "results" in data and isinstance(data["results"], list):
                    return pd.DataFrame(data["results"])
                return pd.DataFrame([data])
    except Exception:
        return None
    return None


def _infer_dataset_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.upper() in {"A", "B", "C", "D"}:
            return part.upper()
    return None


def _infer_method_from_path(path: Path) -> str:
    p = str(path).lower()
    if "baselines_sdv" in p and "ctgan" in p:
        return "SDV-CTGAN"
    if "baselines_sdv" in p and "tvae" in p:
        return "SDV-TVAE"
    if "baselines_sdv" in p and "gaussiancopula" in p:
        return "SDV-GaussianCopula"
    if "dp_diffusion_classcond_dpcr" in p:
        return "DP-Diffusion+DP-CR (ClassCond)"
    if "dp_diffusion_classcond" in p:
        return "DP-Diffusion (ClassCond)"
    if "dp_diffusion_dpcr" in p:
        return "DP-Diffusion+DP-CR"
    if "dp_diffusion" in p:
        return "DP-Diffusion"
    return "UNKNOWN"


def _infer_seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.lower().startswith("seed"):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except Exception:
                    return None
    return None


def _infer_eps_from_path(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = str(path).lower()
    eps_train = None
    eps_repair = None
    m_train = re.search(r"train([0-9p.]+)", p)
    if m_train:
        try:
            eps_train = float(m_train.group(1).replace("p", "."))
        except Exception:
            pass
    m_rep = re.search(r"repair([0-9p.]+)", p)
    if m_rep:
        try:
            eps_repair = float(m_rep.group(1).replace("p", "."))
        except Exception:
            pass
    eps_total = None
    if eps_train is not None and eps_repair is not None:
        eps_total = eps_train + eps_repair
    elif eps_train is not None:
        eps_total = eps_train
    return eps_total, eps_train, eps_repair


def discover_all_synth_runs(outputs_root: Path) -> pd.DataFrame:
    rows = []
    synth_by_dir = {}
    for p in outputs_root.rglob("synth.csv"):
        if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
            continue
        synth_by_dir[p.parent] = p
    for p in outputs_root.rglob("synth_unrepaired.csv"):
        if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
            continue
        if p.parent in synth_by_dir:
            continue
        synth_by_dir[p.parent] = p
    for p in synth_by_dir.values():
        dataset = _infer_dataset_from_path(p)
        method = _infer_method_from_path(p)
        seed = _infer_seed_from_path(p)
        eps_total, eps_train, eps_repair = _infer_eps_from_path(p)
        repaired_flag = np.nan
        if "dp_diffusion_dpcr" in str(p).lower() or "dp_diffusion_classcond_dpcr" in str(p).lower():
            repaired_flag = p.name.lower() == "synth.csv"
        row = {
            "syn_path": str(p),
            "run_dir": str(p.parent),
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "epsilon": eps_total,
            "epsilon_train": eps_train,
            "epsilon_repair": eps_repair,
            "repaired_flag": repaired_flag,
            "syn_path_key": normalize_path(str(p)),
            "run_dir_key": normalize_dir_key(str(p.parent)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def backfill_sdv_tstr_for_dataset(dataset: str, outputs_root: Path) -> pd.DataFrame:
    sdv_root = outputs_root / dataset / "baselines_sdv"
    synth_paths = list(sdv_root.rglob("synth.csv"))
    if not synth_paths:
        return pd.DataFrame()
    rows = []
    for sp in synth_paths:
        method = _infer_method_from_path(sp)
        eps_total, eps_train, eps_repair = _infer_eps_from_path(sp)
        row = evaluate_one(dataset, sp, method, eps_total)
        row["seed"] = _infer_seed_from_path(sp)
        row["epsilon_train"] = eps_train
        row["epsilon_repair"] = eps_repair
        row["syn_path"] = str(sp)
        rows.append(row)
    df_new = pd.DataFrame(rows)
    out_csv = outputs_root / f"tstr_corr_results_{dataset}.csv"
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df_old["syn_path_norm"] = df_old["syn_path"].apply(norm_path) if "syn_path" in df_old.columns else np.nan
        df_new["syn_path_norm"] = df_new["syn_path"].apply(norm_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["syn_path_norm"], keep="last")
        df_all = df_all.drop(columns=["syn_path_norm"], errors="ignore")
        df_all.to_csv(out_csv, index=False)
    else:
        df_new.to_csv(out_csv, index=False)
    return df_new


def _discover_privacy_files(outputs_dir: Path) -> List[Path]:
    files = []
    for ext in ["*.csv", "*.json"]:
        for p in outputs_dir.rglob(ext):
            name = p.name.lower()
            if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
                continue
            if any(k in name for k in PRIVACY_KEYWORDS):
                files.append(p)
    return files


def discover_leakage_audit_table(outputs_root: Path) -> pd.DataFrame:
    rows = []
    for p in outputs_root.rglob("leakage_audit.json"):
        if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        nn_stats_norm = metrics.get("nn_distance_stats_norm", {}) if isinstance(metrics, dict) else {}
        nn_median = None
        if isinstance(nn_stats_norm, dict):
            nn_median = nn_stats_norm.get("median")
        dup_rate = metrics.get("duplicate_rate") if isinstance(metrics, dict) else None
        rows.append(
            {
                "run_dir": str(p.parent),
                "run_dir_key": normalize_dir_key(str(p.parent)),
                "nn_dist_norm_median": nn_median,
                "duplicate_rate": dup_rate,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["nn_dist_norm_median"] = pd.to_numeric(df["nn_dist_norm_median"], errors="coerce")
    bad = df["nn_dist_norm_median"] <= 0
    df.loc[bad, "nn_dist_norm_median"] = np.nan
    df["duplicate_rate"] = pd.to_numeric(df["duplicate_rate"], errors="coerce")
    return df


def discover_knn_mia_table(outputs_root: Path) -> pd.DataFrame:
    preferred = load_preferred_knn(outputs_root)
    if not preferred.empty:
        return preferred

    direct_path = outputs_root / "knn_mia_results.csv"
    candidates = []
    for ext in ["*.csv", "*.json"]:
        for p in outputs_root.rglob(ext):
            name = p.name.lower()
            if name in {"synth.csv", "synth_unrepaired.csv"}:
                continue
            if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
                continue
            if any(k in name for k in PRIVACY_KEYWORDS):
                candidates.append(p)

    rows = []
    for p in candidates:
        df = _read_any_table(p)
        if df is None or df.empty:
            continue
        df = df.copy()

        # Find candidate KNN columns (case-insensitive).
        # Prefer reported knn_mia_auc over knn_mia_auc_raw when both exist.
        cols_lower = {c.lower(): c for c in df.columns}
        if "knn_mia_auc" in cols_lower:
            candidate_cols = [cols_lower["knn_mia_auc"]]
        else:
            candidate_cols = []
            if "knn_mia_auc_raw" in cols_lower:
                candidate_cols.append(cols_lower["knn_mia_auc_raw"])
            for key in KNN_CANDIDATE_COLS:
                if key.lower() in cols_lower:
                    candidate_cols.append(cols_lower[key.lower()])
            if not candidate_cols:
                for c in df.columns:
                    c_l = c.lower()
                    if any(k in c_l for k in KNN_CANDIDATE_COLS):
                        candidate_cols.append(c)
            # Preserve order while deduplicating
            seen = set()
            candidate_cols = [c for c in candidate_cols if not (c in seen or seen.add(c))]

        best_col = None
        best_score = (-1.0, -1.0)
        for c in candidate_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if vals.empty:
                continue
            in_range = vals.between(0, 1).mean()
            var = vals.var()
            non_zero = (vals != 0).sum()
            frac_zero = (vals == 0).mean()
            if in_range < 0.95 or var <= 1e-8:
                continue
            if frac_zero > 0.2 and not (var > 1e-4 and non_zero >= max(10, int(0.5 * len(vals)))):
                continue
            score = (in_range, var)
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is None:
            continue

        vals = pd.to_numeric(df[best_col], errors="coerce")
        v = vals.dropna()
        if v.empty:
            continue
        if v.between(0, 1).mean() < 0.95 or v.nunique(dropna=True) <= 1:
            continue

        if "status" in df.columns:
            bad = df["status"].astype(str) != "OK"
            vals = vals.where(~bad, np.nan)
        if "fail_code" in df.columns:
            bad = df["fail_code"].astype(str).str.len() > 0
            vals = vals.where(~bad, np.nan)
        v2 = vals.dropna()
        if not v2.empty:
            frac_zero = (v2 == 0).mean()
            non_zero = (v2 != 0).sum()
            total = len(v2)
            if frac_zero > 0.2 and non_zero < int(0.5 * total):
                vals = vals.where(vals != 0, np.nan)

        # Determine join keys
        run_dir = None
        syn_path = None
        if "run_dir" in df.columns:
            run_dir = df["run_dir"].astype(str)
        if "syn_path" in df.columns:
            syn_path = df["syn_path"].astype(str)
        elif "synth_path" in df.columns:
            syn_path = df["synth_path"].astype(str)

        if run_dir is None and syn_path is None:
            path_col = None
            for c in df.columns:
                series = df[c].astype(str)
                if series.str.contains("outputs\\\\|outputs/", regex=True).any():
                    path_col = c
                    break
            if path_col:
                paths = df[path_col].astype(str)
                syn_path = paths.where(paths.str.contains("synth\\.csv|synth_unrepaired\\.csv", case=False, regex=True))
                if syn_path.notna().any():
                    run_dir = syn_path.map(lambda s: str(Path(str(s)).parent))
                else:
                    run_dir = paths

        raw_vals = None
        if "knn_mia_auc_raw" in df.columns:
            raw_vals = pd.to_numeric(df["knn_mia_auc_raw"], errors="coerce")

        out = pd.DataFrame({"knn_mia_auc": vals})
        if raw_vals is not None:
            out["knn_mia_auc_raw"] = raw_vals
        if run_dir is not None:
            out["run_dir"] = run_dir.map(normalize_dir_key)
            out["run_dir_key"] = out["run_dir"]
        if syn_path is not None:
            out["syn_path"] = syn_path.map(normalize_path)
            out["syn_path_key"] = out["syn_path"]
        out["_knn_source_file"] = p.as_posix()
        out["_knn_source_col"] = best_col
        out["_mtime"] = p.stat().st_mtime
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    knn_all = pd.concat(rows, ignore_index=True)
    knn_all = knn_all.loc[:, ~knn_all.columns.duplicated()].reset_index(drop=True)
    return knn_all


def load_preferred_knn(outputs_root: Path) -> pd.DataFrame:
    path = outputs_root / "knn_mia_results.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "run_dir_key" not in df.columns:
        if "run_dir" in df.columns:
            df["run_dir_key"] = df["run_dir"].map(normalize_dir_key)
    if "syn_path_key" not in df.columns:
        if "syn_path" in df.columns:
            df["syn_path_key"] = df["syn_path"].map(normalize_path)
        elif "synth_path" in df.columns:
            df["syn_path_key"] = df["synth_path"].map(normalize_path)
    required = [
        "run_dir_key",
        "syn_path_key",
        "knn_mia_auc",
        "knn_mia_auc_raw",
        "auc_flipped",
        "status",
        "fail_code",
        "generated_at",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    df["knn_mia_auc"] = pd.to_numeric(df["knn_mia_auc"], errors="coerce")
    if "status" in df.columns:
        df.loc[df["status"].astype(str) != "OK", "knn_mia_auc"] = np.nan
    df["_knn_source_file"] = path.as_posix()
    df["_knn_source_col"] = "knn_mia_auc"
    df["_mtime"] = path.stat().st_mtime
    return df


def _should_skip_mia_file(path: Path) -> bool:
    name = path.name.lower()
    path_str = path.as_posix().lower()
    return any(pattern in name for pattern in MIA_SKIP_FILE_PATTERNS) or any(
        pattern in path_str for pattern in MIA_SKIP_FILE_PATTERNS
    )


def _filter_mia_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    skip_mask = pd.Series(False, index=df.index)
    check_cols = ["_mia_source_file", "_source_file", "source_file", "file", "path"]
    for col in check_cols:
        if col not in df.columns:
            continue
        col_vals = df[col].astype(str).str.lower()
        for pattern in MIA_SKIP_FILE_PATTERNS:
            skip_mask |= col_vals.str.contains(pattern)
    return df[~skip_mask]


def discover_mia_table(outputs_root: Path) -> pd.DataFrame:
    candidates = []
    for ext in ["*.csv", "*.json"]:
        for p in outputs_root.rglob(ext):
            name = p.name.lower()
        if name in {"synth.csv", "synth_unrepaired.csv"}:
            continue
        if _should_skip_mia_file(p):
            continue
            path_str = p.as_posix().lower()
            if "knn" in name or "/knn_mia/" in path_str or "knn_mia" in path_str:
                continue
            if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
                continue
            if any(k in name for k in ["mia", "attack", "privacy", "audit", "leakage"]):
                candidates.append(p)

    rows = []
    for p in candidates:
        df = _read_any_table(p)
        if df is None or df.empty:
            continue
        df = df.copy()
        df = _filter_mia_source_rows(df)
        if df.empty:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        candidate_cols = []
        # Prefer columns containing both "mia" and "auc"; never select knn-related columns here.
        for key in ["mia_auc", "mia auc", "attack_auc", "mia_auc_star", "mia_auc_raw"]:
            if key.lower() in cols_lower:
                col = cols_lower[key.lower()]
                if "knn" not in col.lower():
                    candidate_cols.append(col)
        if not candidate_cols:
            for c in df.columns:
                c_l = c.lower()
                if "knn" in c_l:
                    continue
                if "mia" in c_l and "auc" in c_l:
                    candidate_cols.append(c)
        if not candidate_cols:
            for c in df.columns:
                c_l = c.lower()
                if "knn" in c_l:
                    continue
                if "auc" in c_l:
                    candidate_cols.append(c)

        best_col = None
        best_score = (-1.0, -1.0)
        for c in candidate_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if vals.empty:
                continue
            if len(vals) == 1:
                in_range = 1.0 if vals.iloc[0] >= 0 and vals.iloc[0] <= 1 else 0.0
                var = 1.0
            else:
                in_range = vals.between(0, 1).mean()
                var = vals.var()
            if in_range < 0.95 or var <= 1e-8:
                continue
            score = (in_range, var)
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is None:
            continue

        vals = pd.to_numeric(df[best_col], errors="coerce")
        v = vals.dropna()
        if v.empty:
            continue
        if len(v) > 1:
            if v.between(0, 1).mean() < 0.95 or v.nunique(dropna=True) <= 1:
                continue

        run_dir = None
        syn_path = None
        if "run_dir" in df.columns:
            run_dir = df["run_dir"].astype(str)
        if "syn_path" in df.columns:
            syn_path = df["syn_path"].astype(str)
        elif "synth_path" in df.columns:
            syn_path = df["synth_path"].astype(str)

        if run_dir is None and syn_path is None:
            path_col = None
            for c in df.columns:
                series = df[c].astype(str)
                if series.str.contains("outputs\\\\|outputs/", regex=True).any():
                    path_col = c
                    break
            if path_col:
                paths = df[path_col].astype(str)
                syn_path = paths.where(paths.str.contains("synth\\.csv|synth_unrepaired\\.csv", case=False, regex=True))
                if syn_path.notna().any():
                    run_dir = syn_path.map(lambda s: str(Path(str(s)).parent))
                else:
                    run_dir = paths

        out = pd.DataFrame({"mia_auc": vals})
        if run_dir is not None:
            out["run_dir"] = run_dir.map(normalize_dir_key)
            out["run_dir_key"] = out["run_dir"]
        if syn_path is not None:
            out["syn_path"] = syn_path.map(normalize_path)
            out["syn_path_key"] = out["syn_path"]
        out["_mia_source_file"] = p.as_posix()
        out["_mia_source_col"] = best_col
        out["_mtime"] = p.stat().st_mtime
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    mia_all = pd.concat(rows, ignore_index=True)
    mia_all = mia_all.loc[:, ~mia_all.columns.duplicated()].reset_index(drop=True)
    return mia_all


def load_mia_synthdep(outputs_root: Path) -> pd.DataFrame:
    path = outputs_root / "mia_synthdep_results.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "run_dir_key" not in df.columns:
        if "run_dir" in df.columns:
            df["run_dir_key"] = df["run_dir"].map(normalize_dir_key)
    if "syn_path_key" not in df.columns:
        if "syn_path" in df.columns:
            df["syn_path_key"] = df["syn_path"].map(normalize_path)
    if "mia_auc_synthdep" in df.columns:
        df["mia_auc_synthdep"] = pd.to_numeric(df["mia_auc_synthdep"], errors="coerce")
    df["_mia_synthdep_source_file"] = path.as_posix()
    return df

def _merge_privacy(master: pd.DataFrame, privacy_files: List[Path]) -> Tuple[pd.DataFrame, List[str]]:
    if not privacy_files:
        return master, []

    privacy_rows = []
    for p in privacy_files:
        df = _read_privacy_file(p)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "run_id_src"})
        if "run_id_alt" in df.columns:
            df = df.rename(columns={"run_id_alt": "run_id_alt_src"})
        if _first_existing_col(df, PATH_COL_CANDIDATES) is None:
            df["path"] = str(p.parent)
        if "dataset" not in df.columns:
            df["dataset"] = df.get("dataset", pd.NA)
        df["run_id"] = _build_run_id(df)
        df["run_id_alt"] = _build_run_id_alt(df)
        df["_source_file"] = p.name
        privacy_rows.append(df)

    if not privacy_rows:
        return master, [str(p) for p in privacy_files]

    privacy_all = pd.concat(privacy_rows, ignore_index=True)
    privacy_all = privacy_all.loc[:, ~privacy_all.columns.duplicated()].reset_index(drop=True)

    def coalesce_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
        rows = []
        for k, g in df.groupby(key, dropna=False):
            row = {key: k}
            for c in g.columns:
                if c == key:
                    continue
                vals = g[c].dropna()
                row[c] = vals.iloc[0] if not vals.empty else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    by_run = coalesce_by_key(privacy_all, "run_id")
    by_alt = coalesce_by_key(privacy_all, "run_id_alt")
    by_run = by_run.loc[:, ~by_run.columns.duplicated()].reset_index(drop=True)
    by_alt = by_alt.loc[:, ~by_alt.columns.duplicated()].reset_index(drop=True)

    merged = master.merge(by_run, on="run_id", how="left", suffixes=("", "_privacy"))

    by_alt = by_alt.rename(
        columns={c: f"{c}_alt" for c in by_alt.columns if c != "run_id_alt"}
    )
    by_alt = by_alt.loc[:, ~by_alt.columns.duplicated()]
    merged = merged.merge(by_alt, on="run_id_alt", how="left")

    # Fill missing privacy fields from alt merge
    alt_cols = [c for c in merged.columns if c.endswith("_alt")]
    for alt in alt_cols:
        base = alt[:-4]
        if base in merged.columns:
            merged[base] = merged[base].combine_first(merged[alt])
    merged = merged.drop(columns=alt_cols, errors="ignore")

    return merged, [str(p) for p in privacy_files]


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _repaired_flag(row: pd.Series) -> Optional[bool]:
    path_col = None
    for c in PATH_COL_CANDIDATES:
        if c in row:
            path_col = c
            break
    if not path_col:
        return np.nan
    p = str(row[path_col]).lower()
    if "synth_unrepaired.csv" in p:
        return False
    if "dp_diffusion_dpcr" in p and "synth.csv" in p:
        return True
    return np.nan


def _load_tstr_results(outputs_dir: Path) -> pd.DataFrame:
    frames = []
    for ds in ["A", "B", "C", "D"]:
        path = outputs_dir / f"tstr_corr_results_{ds}.csv"
        if not path.exists():
            print(f"[WARN] Missing utility file: {path}")
            continue
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_tstr_backfill(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "tstr_corr_backfill_incremental.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "run_dir_key" not in df.columns:
        if "run_dir" in df.columns:
            df["run_dir_key"] = df["run_dir"].astype(str).map(normalize_dir_key)
        elif "syn_path" in df.columns:
            df["run_dir_key"] = df["syn_path"].astype(str).map(normalize_dir_key)
    if "syn_path" in df.columns:
        df["syn_path_key"] = df["syn_path"].astype(str).map(normalize_path)
        df["syn_path_norm"] = df["syn_path"].apply(norm_path)
    if "_mtime" not in df.columns:
        df["_mtime"] = 0.0
    ok_flag = df.get("status", pd.Series([None] * len(df))).astype(str).str.upper() == "OK"
    df["_ok_first"] = ok_flag.astype(int)
    df = df.sort_values(by=["_ok_first", "_mtime"], ascending=[False, False])
    if "run_dir_key" in df.columns:
        df = df.drop_duplicates(subset=["run_dir_key"], keep="first")
    df = df.drop(columns=["_ok_first"], errors="ignore")
    return df


def _merge_tstr_backfill(tstr_df: pd.DataFrame, backfill_df: pd.DataFrame) -> pd.DataFrame:
    if tstr_df.empty:
        return backfill_df
    if backfill_df.empty:
        return tstr_df
    tstr_df = tstr_df.copy()
    backfill_df = backfill_df.copy()
    if "run_dir_key" not in tstr_df.columns:
        if "syn_path" in tstr_df.columns:
            tstr_df["run_dir_key"] = tstr_df["syn_path"].astype(str).map(normalize_dir_key)
        elif "run_dir" in tstr_df.columns:
            tstr_df["run_dir_key"] = tstr_df["run_dir"].astype(str).map(normalize_dir_key)
    if "run_dir_key" not in backfill_df.columns:
        return tstr_df
    base_keys = set(tstr_df["run_dir_key"].dropna().astype(str))
    new_rows = backfill_df[backfill_df["run_dir_key"].astype(str).isin(set(backfill_df["run_dir_key"]) - base_keys)]
    if not new_rows.empty:
        tstr_df = pd.concat([tstr_df, new_rows], ignore_index=True)
    backfill_renamed = backfill_df.rename(
        columns={c: f"{c}_bf" for c in backfill_df.columns if c != "run_dir_key"}
    )
    merged = tstr_df.merge(backfill_renamed, on="run_dir_key", how="left")
    fill_cols = [
        "tstr_acc",
        "tstr_balacc",
        "tstr_f1macro",
        "pearson_mad",
        "corr_status",
        "corr_error",
        "status",
        "fail_code",
        "error_msg",
        "invalid_for_tstr",
    ]
    for col in fill_cols:
        bf = f"{col}_bf"
        if bf not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged[col].combine_first(merged[bf])
    drop_cols = [c for c in merged.columns if c.endswith("_bf")]
    return merged.drop(columns=drop_cols, errors="ignore")



def _parse_maybe_json(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return obj
    return obj


def _extract_nested(obj, keys):
    cur = _parse_maybe_json(obj)
    for key in keys:
        if not isinstance(cur, dict):
            return np.nan
        cur = cur.get(key)
    return cur if cur is not None else np.nan


def _extract_leakage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "metrics" not in df.columns:
        return df

    def get_median(row):
        return _extract_nested(row.get("metrics"), ["nn_distance_stats_norm", "median"])

    def get_dup_rate(row):
        return _extract_nested(row.get("metrics"), ["duplicate_rate"])

    df["nn_dist_norm_median"] = df.apply(get_median, axis=1)
    df["duplicate_rate"] = df.apply(get_dup_rate, axis=1)
    return df


def normalize_privacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _extract_leakage_metrics(df)
    if "membership_audit_adv" not in df.columns and "mia_advantage_abs" in df.columns:
        df = df.rename(columns={"mia_advantage_abs": "membership_audit_adv"})
    if "membership_audit_adv" in df.columns:
        df["membership_audit_adv"] = pd.to_numeric(df["membership_audit_adv"], errors="coerce")
    # mia_auc
    if "mia_auc" not in df.columns or df["mia_auc"].isna().all():
        if "mia_auc_star" in df.columns and df["mia_auc_star"].notna().any():
            df["mia_auc"] = df["mia_auc_star"]
        elif "mia_auc_raw" in df.columns and df["mia_auc_raw"].notna().any():
            df["mia_auc"] = df["mia_auc_raw"]
        elif "mia_advantage_abs" in df.columns and df["mia_advantage_abs"].notna().any():
            df["mia_auc"] = np.nan
    # knn_mia_auc
    if "knn_mia_auc" not in df.columns or df["knn_mia_auc"].isna().all():
        if "knn_mia" in df.columns and df["knn_mia"].notna().any():
            df["knn_mia_auc"] = df["knn_mia"]
        elif "knn_auc" in df.columns and df["knn_auc"].notna().any():
            df["knn_mia_auc"] = df["knn_auc"]
    # leakage_pass
    if "leakage_pass" not in df.columns:
        if "leakage_gate_passed" in df.columns:
            df["leakage_pass"] = df["leakage_gate_passed"]

    df = _ensure_columns(df, PRIVACY_CANONICAL)
    return df


def _summary_mean_std(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "method"], dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
    return summary


def _choose_primary_metric(df: pd.DataFrame) -> str:
    if "tstr_task_type" in df.columns and (df["tstr_task_type"].astype(str) == "regression").any():
        return "tstr_r2" if "tstr_r2" in df.columns else "tstr_acc"
    if "tstr_balacc" in df.columns and df["tstr_balacc"].notna().any():
        return "tstr_balacc"
    return "tstr_acc"


def _pick_winner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    primary = _choose_primary_metric(df)
    for c in [primary, "tstr_f1macro", "tstr_acc", "pearson_mad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if primary in df.columns:
        df = df[df[primary].notna()]
    if df.empty:
        return pd.DataFrame([{"dataset": np.nan}])
    sort_cols = [primary, "tstr_f1macro", "tstr_acc", "pearson_mad"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    return df.iloc[[0]]


def _write_fig(path: Path):
    fig = plt.gcf()
    savefig_pdf_png(fig, path, None)


def _short_method_label(name: str) -> str:
    label = str(name or "")
    if label.startswith("SDV-"):
        label = label.replace("SDV-", "", 1)
    return label.replace("GaussianCopula", "Gaussian Copula")


def plot_privacy_forest_by_dataset(summary_df: pd.DataFrame, metric: str, out_path: Path, xlabel: str):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return
    if summary_df[mean_col].notna().sum() == 0:
        return
    methods = order_methods(summary_df["method"].astype(str).unique().tolist())
    datasets = order_datasets(summary_df["dataset"].astype(str).unique().tolist())
    if not datasets:
        return

    height = 3.6 if len(datasets) > 1 else 2.4
    fig, axes = plt.subplots(len(datasets), 1, figsize=paper_figsize(height), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = summary_df[summary_df["dataset"].astype(str) == ds].copy()
        y_pos = np.arange(len(methods))
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_short_method_label(m) for m in methods])
        for i, m in enumerate(methods):
            row = sub[sub["method"].astype(str) == m]
            if row.empty:
                continue
            mean = row.iloc[0][mean_col]
            std = row.iloc[0][std_col]
            if pd.isna(mean):
                continue
            ax.errorbar(
                float(mean),
                i,
                xerr=float(std) if pd.notna(std) else 0.0,
                fmt="o",
                markersize=5,
                capsize=2.5,
                elinewidth=1.0,
                color=method_color_paper(m),
            )
        ax.set_title(f"Dataset {ds}", fontsize=9, loc="left", pad=2)
        ax.axvline(0.5, linestyle="--", linewidth=0.8, alpha=0.4, color="gray")
        ax.set_xlim(0.0, 0.5)
        style_axes(ax)
    axes[-1].set_xlabel(xlabel)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.12)
    savefig_pdf_png(fig, out_path, None)


def plot_corr_forest(summary_df: pd.DataFrame, metric: str, out_path: Path, xlabel: str):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return
    if summary_df[mean_col].notna().sum() == 0:
        return
    df_plot = summary_df.copy()
    df_plot[mean_col] = pd.to_numeric(df_plot[mean_col], errors="coerce")
    df_plot[std_col] = pd.to_numeric(df_plot[std_col], errors="coerce")
    df_plot = df_plot[df_plot[mean_col].notna()].copy()
    if df_plot.empty:
        return

    methods = order_methods(df_plot["method"].astype(str).unique().tolist())
    means = []
    stds = []
    labels = []
    colors = []
    for m in methods:
        row = df_plot[df_plot["method"].astype(str) == m].iloc[0]
        means.append(float(row[mean_col]))
        stds.append(float(row[std_col]) if pd.notna(row[std_col]) else 0.0)
        labels.append(_short_method_label(m))
        colors.append(method_color_paper(m))

    order = np.argsort(means)
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=paper_figsize(2.4))
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=stds, fmt="o", markersize=5, capsize=2.5, elinewidth=1.0, color="black")
    for i, c in enumerate(colors):
        ax.scatter(means[i], y[i], s=30, color=c, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    style_axes(ax)
    fig.subplots_adjust(left=0.45, right=0.98, top=0.95, bottom=0.20)
    savefig_pdf_png(fig, out_path, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--outputs_dir", type=str, default="", help="Deprecated alias for --outputs_root")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--paper_dir", type=str, default="")
    parser.add_argument("--overwrite", type=int, default=1)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_dir) if args.outputs_dir else Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else outputs_root
    paper_dir = Path(args.paper_dir) if args.paper_dir else (out_dir / "paper_package")
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    base_df = discover_all_synth_runs(outputs_root)
    print(f"[AUDIT] discovered_synth_rows={len(base_df)}")
    if "syn_path_key" in base_df.columns:
        print(f"[AUDIT] unique_syn_path_key={base_df['syn_path_key'].nunique()}")

    tstr_df = _load_tstr_results(outputs_root)
    if tstr_df.empty:
        print("[WARN] No tstr_corr results found; master will contain synth runs only.")

    keep = [d.strip().upper() for d in args.datasets.split(",") if d.strip()] if args.datasets else []
    if keep:
        base_df = base_df[base_df["dataset"].astype(str).str.upper().isin(keep)].copy()
        tstr_df = tstr_df[tstr_df["dataset"].astype(str).str.upper().isin(keep)].copy()
        print(f"[STAT] dataset_filter={keep} base_rows={len(base_df)} tstr_rows={len(tstr_df)}")
    tstr_df["run_id"] = _build_run_id(tstr_df)
    tstr_df["run_id_alt"] = _build_run_id_alt(tstr_df)
    if "syn_path" in tstr_df.columns:
        tstr_df["syn_path_key"] = tstr_df["syn_path"].astype(str).map(normalize_path)
        tstr_df["run_dir_key"] = tstr_df["syn_path"].astype(str).map(normalize_dir_key)
        tstr_df["syn_path_norm"] = tstr_df["syn_path"].apply(norm_path)
    elif "synth_path" in tstr_df.columns:
        tstr_df["syn_path_key"] = tstr_df["synth_path"].astype(str).map(normalize_path)
        tstr_df["run_dir_key"] = tstr_df["synth_path"].astype(str).map(normalize_dir_key)
        tstr_df["syn_path_norm"] = tstr_df["synth_path"].apply(norm_path)
    tstr_df["repaired_flag"] = tstr_df.apply(_repaired_flag, axis=1)
    tstr_backfill = _load_tstr_backfill(outputs_root)
    if keep:
        tstr_backfill = tstr_backfill[tstr_backfill["dataset"].astype(str).str.upper().isin(keep)].copy()
    if not tstr_backfill.empty:
        tstr_df = _merge_tstr_backfill(tstr_df, tstr_backfill)

    tstr_df = _ensure_columns(
        tstr_df,
        ["dataset", "method", "seed", "status", "run_id", "run_id_alt", "syn_path_key", "run_dir_key"]
        + UTILITY_CANONICAL
        + ["tstr_balacc_cat", "tstr_f1macro_cat", "rtr_balacc_cat", "rtr_f1macro_cat"]
        + ["epsilon_train", "epsilon_repair", "repaired_flag", "invalid_for_tstr"],
    )

    privacy_files = _discover_privacy_files(outputs_root)
    if not privacy_files:
        warnings.append("No privacy files found.")
    master_df, privacy_used = _merge_privacy(tstr_df, privacy_files)
    if not privacy_used and privacy_files:
        warnings.append("Privacy files found but no usable rows were parsed.")
    # Expand master with all synth runs
    utility_merge_by_syn_path = 0
    utility_merge_by_run_dir = 0
    if not base_df.empty:
        base_df = base_df.loc[:, ~base_df.columns.duplicated()].reset_index(drop=True)
        if "syn_path" in base_df.columns:
            base_df["syn_path_norm"] = base_df["syn_path"].apply(norm_path)
        tstr_merge = tstr_df.copy()
        tstr_merge = tstr_merge.loc[:, ~tstr_merge.columns.duplicated()]
        master_df = base_df.merge(
            tstr_merge.drop(columns=["run_dir", "syn_path"], errors="ignore"),
            on="syn_path_norm",
            how="left",
            suffixes=("", "_tstr"),
            indicator=True,
        )
        utility_merge_by_syn_path = int((master_df["_merge"] == "both").sum())
        master_df = master_df.drop(columns=["_merge"], errors="ignore")

        if "run_dir_key" in base_df.columns and "run_dir_key" in tstr_merge.columns:
            tstr_merge_run = tstr_merge.drop_duplicates(subset=["run_dir_key"])
            fallback = base_df.merge(
                tstr_merge_run.drop(columns=["run_dir", "syn_path"], errors="ignore"),
                on="run_dir_key",
                how="left",
                suffixes=("", "_tstr2"),
                indicator=True,
            )
            missing_mask = master_df[["tstr_balacc", "tstr_acc", "tstr_f1macro"]].isna().all(axis=1)
            used_mask = missing_mask & (fallback["_merge"] == "both")
            utility_merge_by_run_dir = int(used_mask.sum())
            for col in fallback.columns:
                if not col.endswith("_tstr2"):
                    continue
                base_col = col[: -len("_tstr2")]
                if base_col in master_df.columns:
                    master_df.loc[used_mask, base_col] = master_df.loc[used_mask, base_col].combine_first(
                        fallback.loc[used_mask, col]
                    )
            fallback = fallback.drop(columns=["_merge"], errors="ignore")
    else:
        master_df = tstr_df.copy()

    print(f"[AUDIT] utility_merge_by_syn_path={utility_merge_by_syn_path}")
    print(f"[AUDIT] utility_merge_by_run_dir={utility_merge_by_run_dir}")

    # Merge MIA table (two-pass)
    mia_table = discover_mia_table(outputs_root)
    if not mia_table.empty:
        mia_table = mia_table.loc[:, ~mia_table.columns.duplicated()].copy()
        for c in ["run_dir", "syn_path"]:
            if c in mia_table.columns:
                if c == "run_dir":
                    mia_table[c] = mia_table[c].map(normalize_dir_key)
                else:
                    mia_table[c] = mia_table[c].map(normalize_path)

        def _pick_best(df: pd.DataFrame, key: str) -> pd.DataFrame:
            rows = []
            for k, g in df.groupby(key, dropna=False):
                if pd.isna(k):
                    continue
                g = g.copy()
                g["_non_null"] = g.notna().sum(axis=1)
                g = g.sort_values(by=["_non_null", "_mtime"], ascending=[False, False])
                rows.append(g.iloc[0])
            return pd.DataFrame(rows).drop(columns=["_non_null"], errors="ignore")

        mia_by_run = _pick_best(mia_table, "run_dir_key") if "run_dir_key" in mia_table.columns else pd.DataFrame()
        mia_by_syn = _pick_best(mia_table, "syn_path_key") if "syn_path_key" in mia_table.columns else pd.DataFrame()

        if "run_dir_key" in master_df.columns and not mia_by_run.empty:
            mia_cols = [c for c in ["mia_auc", "_mia_source_file", "_mia_source_col"] if c in mia_by_run.columns]
            master_df = master_df.merge(
                mia_by_run[["run_dir_key"] + mia_cols].dropna(subset=["mia_auc"]),
                on="run_dir_key",
                how="left",
                suffixes=("", "_mia"),
            )
            if "mia_auc_mia" in master_df.columns:
                master_df["mia_auc"] = master_df["mia_auc"].combine_first(master_df["mia_auc_mia"])
            if "_mia_source_file_mia" in master_df.columns:
                master_df["_mia_source_file"] = master_df.get("_mia_source_file").combine_first(
                    master_df["_mia_source_file_mia"]
                )
            if "_mia_source_col_mia" in master_df.columns:
                master_df["_mia_source_col"] = master_df.get("_mia_source_col").combine_first(
                    master_df["_mia_source_col_mia"]
                )
            master_df = master_df.drop(
                columns=["mia_auc_mia", "_mia_source_file_mia", "_mia_source_col_mia"], errors="ignore"
            )
        if "syn_path_key" in master_df.columns and not mia_by_syn.empty:
            mia_cols = [c for c in ["mia_auc", "_mia_source_file", "_mia_source_col"] if c in mia_by_syn.columns]
            master_df = master_df.merge(
                mia_by_syn[["syn_path_key"] + mia_cols].dropna(subset=["mia_auc"]),
                on="syn_path_key",
                how="left",
                suffixes=("", "_mia2"),
            )
            if "mia_auc_mia2" in master_df.columns:
                master_df["mia_auc"] = master_df["mia_auc"].combine_first(master_df["mia_auc_mia2"])
            if "_mia_source_file_mia2" in master_df.columns:
                master_df["_mia_source_file"] = master_df.get("_mia_source_file").combine_first(
                    master_df["_mia_source_file_mia2"]
                )
            if "_mia_source_col_mia2" in master_df.columns:
                master_df["_mia_source_col"] = master_df.get("_mia_source_col").combine_first(
                    master_df["_mia_source_col_mia2"]
                )
            master_df = master_df.drop(
                columns=["mia_auc_mia2", "_mia_source_file_mia2", "_mia_source_col_mia2"], errors="ignore"
            )

    # Merge synth-dependent MIA and override mia_auc
    mia_synthdep = load_mia_synthdep(outputs_root)
    if not mia_synthdep.empty:
        mia_cols = [c for c in ["mia_auc_synthdep", "_mia_synthdep_source_file"] if c in mia_synthdep.columns]
        if "run_dir_key" in master_df.columns and "run_dir_key" in mia_synthdep.columns:
            master_df = master_df.merge(
                mia_synthdep[["run_dir_key"] + mia_cols].dropna(subset=["mia_auc_synthdep"]),
                on="run_dir_key",
                how="left",
                suffixes=("", "_synthdep"),
            )
        if "syn_path_key" in master_df.columns and "syn_path_key" in mia_synthdep.columns:
            master_df = master_df.merge(
                mia_synthdep[["syn_path_key"] + mia_cols].dropna(subset=["mia_auc_synthdep"]),
                on="syn_path_key",
                how="left",
                suffixes=("", "_synthdep2"),
            )
        if "mia_auc_legacy" not in master_df.columns:
            master_df["mia_auc_legacy"] = master_df.get("mia_auc")
        if "mia_auc_synthdep" in master_df.columns:
            master_df["mia_auc"] = master_df["mia_auc_synthdep"].combine_first(master_df.get("mia_auc"))
        if "mia_auc_synthdep_synthdep" in master_df.columns:
            master_df["mia_auc"] = master_df["mia_auc_synthdep_synthdep"].combine_first(master_df["mia_auc"])
        if "mia_auc_synthdep_synthdep2" in master_df.columns:
            master_df["mia_auc"] = master_df["mia_auc_synthdep_synthdep2"].combine_first(master_df["mia_auc"])
        master_df = master_df.drop(
            columns=[
                "mia_auc_synthdep_synthdep",
                "mia_auc_synthdep_synthdep2",
            ],
            errors="ignore",
        )

    if "tstr_balacc" in master_df.columns and "tstr_balacc_cat" in master_df.columns:
        b_mask = master_df.get("dataset").astype(str).str.upper() == "B"
        if b_mask.any():
            bal = pd.to_numeric(master_df.loc[b_mask, "tstr_balacc"], errors="coerce")
            bal_cat = pd.to_numeric(master_df.loc[b_mask, "tstr_balacc_cat"], errors="coerce")
            use_cat = bal_cat.notna() & (bal.isna() | (bal_cat >= bal))
            utility_main = bal.where(~use_cat, bal_cat)
            master_df.loc[b_mask, "utility_main"] = utility_main
            master_df.loc[b_mask, "utility_main_source"] = np.where(use_cat, "catboost", "sklearn")
            if use_cat.any():
                print("[AUDIT] B utility_main source = 'catboost'")
            else:
                print("[AUDIT] B utility_main source = 'sklearn'")

    if "tstr_balacc" in master_df.columns:
        b_sdv = master_df[
            (master_df.get("dataset").astype(str).str.upper() == "B")
            & (master_df.get("method").astype(str).str.contains("SDV", case=False, na=False))
        ]
        if not b_sdv.empty and b_sdv["tstr_balacc"].isna().any():
            missing = int(b_sdv["tstr_balacc"].isna().sum())
            print(f"[WARN] SDV tstr_balacc missing for dataset B rows: {missing}")
            df_new = backfill_sdv_tstr_for_dataset("B", outputs_root)
            print(f"[INFO] SDV B backfill evaluated={len(df_new)}")
            if not df_new.empty:
                tstr_df = _load_tstr_results(outputs_root)
                if "syn_path" in tstr_df.columns:
                    tstr_df["syn_path_key"] = tstr_df["syn_path"].astype(str).map(normalize_path)
                    tstr_df["run_dir_key"] = tstr_df["syn_path"].astype(str).map(normalize_dir_key)
                    tstr_df["syn_path_norm"] = tstr_df["syn_path"].apply(norm_path)
                elif "synth_path" in tstr_df.columns:
                    tstr_df["syn_path_key"] = tstr_df["synth_path"].astype(str).map(normalize_path)
                    tstr_df["run_dir_key"] = tstr_df["synth_path"].astype(str).map(normalize_dir_key)
                    tstr_df["syn_path_norm"] = tstr_df["synth_path"].apply(norm_path)
                tstr_df = tstr_df.loc[:, ~tstr_df.columns.duplicated()]
                if not base_df.empty:
                    base_df = base_df.loc[:, ~base_df.columns.duplicated()].reset_index(drop=True)
                    if "syn_path" in base_df.columns:
                        base_df["syn_path_norm"] = base_df["syn_path"].apply(norm_path)
                    tstr_merge = tstr_df.copy().loc[:, ~tstr_df.columns.duplicated()]
                    master_df = base_df.merge(
                        tstr_merge.drop(columns=["run_dir", "syn_path"], errors="ignore"),
                        on="syn_path_norm",
                        how="left",
                        suffixes=("", "_tstr"),
                        indicator=False,
                    )
                else:
                    master_df = tstr_df.copy()
            b_sdv = master_df[
                (master_df.get("dataset").astype(str).str.upper() == "B")
                & (master_df.get("method").astype(str).str.contains("SDV", case=False, na=False))
            ]
            missing_post = int(b_sdv["tstr_balacc"].isna().sum()) if not b_sdv.empty else 0
            print(f"[INFO] SDV B missing after backfill={missing_post}")
            if missing_post:
                mask = b_sdv["tstr_balacc"].isna()
                idx = b_sdv[mask].index
                master_df.loc[idx, "status"] = "FAIL"
                master_df.loc[idx, "invalid_for_tstr"] = True

    # Merge KNN-MIA from preferred artifacts only (if present)
    knn_table = discover_knn_mia_table(outputs_root)
    by_run = pd.DataFrame()
    by_syn = pd.DataFrame()
    if not knn_table.empty:
        # Drop any previously merged knn columns to avoid contamination
        drop_cols = [
            "knn_mia_auc",
            "knn_mia_auc_raw",
            "auc_flipped",
            "_knn_source_file",
            "_knn_source_col",
        ]
        master_df = master_df.drop(columns=[c for c in drop_cols if c in master_df.columns], errors="ignore")

        knn_table = knn_table.loc[:, ~knn_table.columns.duplicated()].copy()
        for c in ["run_dir", "syn_path"]:
            if c in knn_table.columns:
                knn_table[c] = knn_table[c].map(normalize_path)

        def _pick_best(df: pd.DataFrame, key: str) -> pd.DataFrame:
            rows = []
            for k, g in df.groupby(key, dropna=False):
                if pd.isna(k):
                    continue
                g = g.copy()
                g["_non_null"] = g.notna().sum(axis=1)
                g = g.sort_values(by=["_non_null", "_mtime"], ascending=[False, False])
                rows.append(g.iloc[0])
            return pd.DataFrame(rows).drop(columns=["_non_null"], errors="ignore")

        by_run = _pick_best(knn_table, "run_dir_key") if "run_dir_key" in knn_table.columns else pd.DataFrame()
        by_syn = _pick_best(knn_table, "syn_path_key") if "syn_path_key" in knn_table.columns else pd.DataFrame()

        if "run_dir_key" in knn_table.columns:
            dup_run = int(knn_table["run_dir_key"].duplicated().sum())
            print(f"[AUDIT] knn_table run_dir_key duplicates={dup_run}")
        if "syn_path_key" in knn_table.columns:
            dup_syn = int(knn_table["syn_path_key"].duplicated().sum())
            print(f"[AUDIT] knn_table syn_path_key duplicates={dup_syn}")

        if "run_dir_key" in master_df.columns and not by_run.empty:
            knn_cols = [
                c
                for c in ["knn_mia_auc", "knn_mia_auc_raw", "_knn_source_file", "_knn_source_col"]
                if c in by_run.columns
            ]
            master_df = master_df.merge(
                by_run[["run_dir_key"] + knn_cols].dropna(subset=["knn_mia_auc"]),
                on="run_dir_key",
                how="left",
                suffixes=("", "_knn"),
            )
            if "knn_mia_auc_knn" in master_df.columns:
                master_df["knn_mia_auc"] = master_df["knn_mia_auc"].combine_first(master_df["knn_mia_auc_knn"])
            if "knn_mia_auc_raw_knn" in master_df.columns:
                master_df["knn_mia_auc_raw"] = master_df.get("knn_mia_auc_raw").combine_first(
                    master_df["knn_mia_auc_raw_knn"]
                )
            if "_knn_source_file_knn" in master_df.columns:
                master_df["_knn_source_file"] = master_df.get("_knn_source_file").combine_first(
                    master_df["_knn_source_file_knn"]
                )
            if "_knn_source_col_knn" in master_df.columns:
                master_df["_knn_source_col"] = master_df.get("_knn_source_col").combine_first(
                    master_df["_knn_source_col_knn"]
                )
            master_df = master_df.drop(
                columns=[
                    "knn_mia_auc_knn",
                    "knn_mia_auc_raw_knn",
                    "_knn_source_file_knn",
                    "_knn_source_col_knn",
                ],
                errors="ignore",
            )

        if "syn_path_key" in master_df.columns and not by_syn.empty:
            knn_cols = [
                c
                for c in ["knn_mia_auc", "knn_mia_auc_raw", "_knn_source_file", "_knn_source_col"]
                if c in by_syn.columns
            ]
            master_df = master_df.merge(
                by_syn[["syn_path_key"] + knn_cols].dropna(subset=["knn_mia_auc"]),
                on="syn_path_key",
                how="left",
                suffixes=("", "_knn2"),
            )
            if "knn_mia_auc_knn2" in master_df.columns:
                master_df["knn_mia_auc"] = master_df["knn_mia_auc"].combine_first(master_df["knn_mia_auc_knn2"])
            if "knn_mia_auc_raw_knn2" in master_df.columns:
                master_df["knn_mia_auc_raw"] = master_df.get("knn_mia_auc_raw").combine_first(
                    master_df["knn_mia_auc_raw_knn2"]
                )
            if "_knn_source_file_knn2" in master_df.columns:
                master_df["_knn_source_file"] = master_df.get("_knn_source_file").combine_first(
                    master_df["_knn_source_file_knn2"]
                )
            if "_knn_source_col_knn2" in master_df.columns:
                master_df["_knn_source_col"] = master_df.get("_knn_source_col").combine_first(
                    master_df["_knn_source_col_knn2"]
                )
            master_df = master_df.drop(
                columns=[
                    "knn_mia_auc_knn2",
                    "knn_mia_auc_raw_knn2",
                    "_knn_source_file_knn2",
                    "_knn_source_col_knn2",
                ],
                errors="ignore",
            )

    leakage_table = discover_leakage_audit_table(outputs_root)
    baseline_leak_path = outputs_root / "leakage_audit_baselines.csv"
    baseline_leak_table = pd.read_csv(baseline_leak_path) if baseline_leak_path.exists() else pd.DataFrame()
    if not leakage_table.empty or not baseline_leak_table.empty:
        if "run_dir_key" not in master_df.columns:
            master_df["run_dir_key"] = master_df.get("run_dir", "").astype(str).map(normalize_dir_key)
        if not leakage_table.empty:
            master_df = master_df.merge(
                leakage_table,
                on="run_dir_key",
                how="left",
                suffixes=("", "_leak"),
            )
            for col in ["nn_dist_norm_median", "duplicate_rate"]:
                leak_col = f"{col}_leak"
                if leak_col in master_df.columns:
                    master_df[col] = master_df.get(col).combine_first(master_df[leak_col])
            master_df = master_df.drop(columns=["nn_dist_norm_median_leak", "duplicate_rate_leak"], errors="ignore")
        if not baseline_leak_table.empty:
            base_df = baseline_leak_table.copy()
            base_df["run_dir_key"] = base_df.get("run_dir", "").astype(str).map(normalize_dir_key)
            if "syn_path" in base_df.columns:
                base_df["syn_path_key"] = base_df["syn_path"].astype(str).map(normalize_path)
            base_df = base_df.dropna(subset=["run_dir_key"])
            baseline_cols = ["run_dir_key", "nn_dist_norm_median", "nn_dist_norm_median_log10", "privacy_adv_primary"]
            merge_cols = [col for col in baseline_cols if col in base_df.columns]
            if len(merge_cols) > 1:
                base_df = base_df.drop_duplicates(subset=["run_dir_key"])
                master_df = master_df.merge(
                    base_df[merge_cols],
                    on="run_dir_key",
                    how="left",
                    suffixes=("", "_baseline"),
                )
                for col in ["nn_dist_norm_median", "nn_dist_norm_median_log10", "privacy_adv_primary"]:
                    baseline_col = f"{col}_baseline"
                    if baseline_col in master_df.columns:
                        master_df[col] = master_df.get(col).combine_first(master_df[baseline_col])
                master_df = master_df.drop(
                    columns=[col for col in master_df.columns if col.endswith("_baseline")],
                    errors="ignore",
                )
        nn_vals = pd.to_numeric(master_df.get("nn_dist_norm_median"), errors="coerce")
        master_df["nn_dist_norm_median_log10"] = np.where(nn_vals > 0, np.log10(nn_vals), np.nan)

        for ds in ["A", "B", "C"]:
            ds_mask = master_df.get("dataset", "").astype(str).str.upper() == ds
            count = int(master_df.loc[ds_mask, "nn_dist_norm_median"].notna().sum())
            print(f"[AUDIT] nn_dist_norm_median_non_nan {ds}={count}")
        b_rows = master_df[(master_df.get("dataset", "").astype(str).str.upper() == "B") & master_df["nn_dist_norm_median"].notna()]
        if not b_rows.empty:
            examples = b_rows["run_dir"].dropna().astype(str).unique().tolist()[:3]
            print("[AUDIT] B leakage_audit merge examples:")
            for ex in examples:
                print(f"  {ex}")

    master_df = normalize_privacy_columns(master_df)
    master_df = ensure_advantage_columns(master_df)
    master_df["privacy_adv_primary"] = choose_primary_privacy_adv(master_df)
    mia_auc_non_nan = int(pd.to_numeric(master_df.get("mia_auc", pd.Series(dtype=float)), errors="coerce").notna().sum())
    knn_auc_non_nan = int(pd.to_numeric(master_df.get("knn_mia_auc", pd.Series(dtype=float)), errors="coerce").notna().sum())
    mia_adv_non_nan = int(pd.to_numeric(master_df.get("mia_adv_fix", pd.Series(dtype=float)), errors="coerce").notna().sum())
    knn_adv_non_nan = int(pd.to_numeric(master_df.get("knn_mia_adv_fix", pd.Series(dtype=float)), errors="coerce").notna().sum())
    privacy_non_nan = int(pd.to_numeric(master_df.get("privacy_adv_primary", pd.Series(dtype=float)), errors="coerce").notna().sum())
    print(
        f"[STAT] mia_auc_non_nan={mia_auc_non_nan} knn_mia_auc_non_nan={knn_auc_non_nan} "
        f"mia_adv_fix_non_nan={mia_adv_non_nan} knn_mia_adv_fix_non_nan={knn_adv_non_nan} "
        f"privacy_adv_primary_non_nan={privacy_non_nan}"
    )
    priv_vals = pd.to_numeric(master_df.get("privacy_adv_primary", pd.Series(dtype=float)), errors="coerce")
    if priv_vals.notna().any():
        print(
            f"[STAT] privacy_adv_primary min={priv_vals.min():.6f} "
            f"med={priv_vals.median():.6f} max={priv_vals.max():.6f}"
        )
    if priv_vals.notna().any() and float(priv_vals.median()) == 0.5:
        print("[ALERT] privacy metric saturated at 0.5; check eval leakage or parsing.")
    if "mia_auc_invalid" in master_df.columns and master_df["mia_auc_invalid"].any():
        warnings.append(f"mia_auc_invalid={int(master_df['mia_auc_invalid'].sum())}")
    if "knn_mia_auc_invalid" in master_df.columns and master_df["knn_mia_auc_invalid"].any():
        warnings.append(f"knn_mia_auc_invalid={int(master_df['knn_mia_auc_invalid'].sum())}")

    # Deduplicate by synth path key
    if "syn_path_key" in master_df.columns:
        master_df["_non_null"] = master_df.notna().sum(axis=1)
        master_df = master_df.sort_values(by="_non_null", ascending=False)
        master_df = master_df.drop_duplicates(subset=["syn_path_key"], keep="first").drop(columns=["_non_null"])

    master_path = out_dir / "master_results.csv"
    master_df.to_csv(master_path, index=False)
    print(f"[OK] Wrote master table: {master_path}")

    # Copy master into paper package
    master_copy = paper_dir / "master_results.csv"
    master_df.to_csv(master_copy, index=False)

    # Summary tables
    summary_metrics = [m for m in ["tstr_balacc", "tstr_acc", "pearson_mad"] if m in master_df.columns]
    privacy_metrics = [
        m
        for m in [
            "mia_adv_fix",
            "knn_mia_adv_fix",
            "privacy_adv_primary",
            "nn_dist_norm_median",
            "mia_auc",
            "knn_mia_auc",
        ]
        if m in master_df.columns
    ]
    summary_df = _summary_mean_std(master_df, summary_metrics + privacy_metrics)
    summary_path = tables_dir / "summary_mean_std.csv"
    summary_df.to_csv(summary_path, index=False)

    # Winners overall
    winners = []
    winners_dp = []
    df_ok = master_df.copy()
    if "status" in df_ok.columns:
        df_ok = df_ok[df_ok["status"].astype(str) == "OK"]
    for ds in ["A", "B", "C", "D"]:
        sub = df_ok[df_ok["dataset"].astype(str).str.upper() == ds]
        if sub.empty:
            winners.append(pd.Series({"dataset": ds, "method": "NONE"}))
            winners_dp.append(pd.Series({"dataset": ds, "method": "NONE"}))
            continue
        winners.append(_pick_winner(sub).iloc[0])
        sub_dp = sub[sub["method"].astype(str).str.contains("DP-", case=False, na=False)]
        if sub_dp.empty:
            winners_dp.append(pd.Series({"dataset": ds, "method": "NONE"}))
        else:
            winners_dp.append(_pick_winner(sub_dp).iloc[0])

    winners_df = pd.DataFrame(winners)
    winners_dp_df = pd.DataFrame(winners_dp)
    for df in [winners_df, winners_dp_df]:
        if "method" in df.columns:
            df.rename(columns={"method": "winner_method"}, inplace=True)
    keep_cols = [
        "dataset",
        "winner_method",
        "epsilon",
        "seed",
        "tstr_acc",
        "tstr_balacc",
        "tstr_f1macro",
        "pearson_mad",
        "mia_adv_fix",
        "knn_mia_adv_fix",
        "privacy_adv_primary",
        "mia_auc",
        "knn_mia_auc",
        "delta_presence",
        "identifiability",
        "leakage_score",
        "leakage_pass",
    ]
    winners_df = winners_df[[c for c in keep_cols if c in winners_df.columns]]
    winners_dp_df = winners_dp_df[[c for c in keep_cols if c in winners_dp_df.columns]]
    winners_df.to_csv(tables_dir / "winners_overall.csv", index=False)
    winners_dp_df.to_csv(tables_dir / "winners_dp_only.csv", index=False)

    print("[INFO] DP-CR ablation artifacts are generated in make_dpcr_ablation_artifacts.py")

    # KNN failures summary (optional)
    knn_fail_path = out_dir / "knn_mia" / "knn_mia_results.csv"
    if knn_fail_path.exists():
        knn_df = pd.read_csv(knn_fail_path)
        if "status" in knn_df.columns and "fail_code" in knn_df.columns:
            fail_counts = (
                knn_df[knn_df["status"].astype(str) != "OK"]["fail_code"]
                .value_counts(dropna=False)
                .reset_index()
            )
            fail_counts.columns = ["fail_code", "count"]
            fail_counts.to_csv(tables_dir / "knn_failures_summary.csv", index=False)

    # Figures
    set_paper_style()

    def _bar_with_err(metric: str, title: str, out_name: str, ylabel: str, ylim: tuple[float, float] | None = None):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary_df.columns or std_col not in summary_df.columns:
            warnings.append(f"Missing mean/std for plot: {metric}")
            return
        df_plot = summary_df.copy()
        methods = order_methods(df_plot["method"].astype(str).unique().tolist())
        datasets = df_plot["dataset"].astype(str).unique().tolist()
        dataset_order = order_datasets(datasets)
        x = np.arange(len(dataset_order))
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        for m_i, m in enumerate(methods):
            sub = df_plot[df_plot["method"].astype(str) == m].set_index("dataset")
            for d_i, ds in enumerate(dataset_order):
                pos = x[d_i] + (m_i - (len(methods) - 1) / 2) * width
                y = sub.loc[ds, mean_col] if ds in sub.index else np.nan
                yerr = sub.loc[ds, std_col] if ds in sub.index else np.nan
                color = method_color(m)
                style = dataset_bar_style(ds, color)
                ax.bar(pos, y, width=width, yerr=yerr, capsize=3, **style)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        method_handles = [
            Patch(facecolor=method_color(m), edgecolor=method_color(m), label=m) for m in methods
        ]
        ax.legend(
            handles=method_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            title="Method",
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        _write_fig(figures_dir / out_name)

    _bar_with_err(
        "tstr_balacc",
        "TSTR Balanced Accuracy (mean +/- std)",
        "fig_utility_tstr_balacc_bar.pdf",
        "TSTR balanced accuracy (higher is better)",
    )
    _bar_with_err(
        "tstr_acc",
        "TSTR Accuracy (mean +/- std)",
        "fig_utility_tstr_acc_bar.pdf",
        "TSTR accuracy (higher is better)",
    )
    plot_corr_forest(
        summary_df,
        "pearson_mad",
        figures_dir / "fig_corr_pearson_mad_bar.pdf",
        "Pearson MAD (down)",
    )

    # Privacy bar
    if "mia_adv_fix_mean" in summary_df.columns and summary_df["mia_adv_fix_mean"].notna().any():
        plot_privacy_forest_by_dataset(
            summary_df,
            "mia_adv_fix",
            figures_dir / "fig_privacy_mia_adv_bar.pdf",
            "Attack advantage |AUC-0.5| (down)",
        )
        src = figures_dir / "fig_privacy_mia_adv_bar.pdf"
        dst = figures_dir / "fig_privacy_mia_auc_bar.pdf"
        if src.exists():
            shutil.copy2(src, dst)
    else:
        warnings.append("mia_adv missing or empty; skipping fig_privacy_mia_adv_bar.pdf")

    if "knn_mia_adv_fix_mean" in summary_df.columns and summary_df["knn_mia_adv_fix_mean"].notna().any():
        plot_privacy_forest_by_dataset(
            summary_df,
            "knn_mia_adv_fix",
            figures_dir / "fig_privacy_knn_mia_adv_bar.pdf",
            "Attack advantage |AUC-0.5| (down)",
        )
        src = figures_dir / "fig_privacy_knn_mia_adv_bar.pdf"
        dst = figures_dir / "fig_privacy_knn_mia_auc_bar.pdf"
        if src.exists():
            shutil.copy2(src, dst)
    else:
        warnings.append("knn_mia_adv missing or empty; skipping fig_privacy_knn_mia_adv_bar.pdf")

    # Frontier scatter
    x_col = None
    if "privacy_adv_primary_mean" in summary_df.columns and summary_df["privacy_adv_primary_mean"].notna().any():
        x_col = "privacy_adv_primary_mean"
    elif "knn_mia_adv_fix_mean" in summary_df.columns and summary_df["knn_mia_adv_fix_mean"].notna().any():
        x_col = "knn_mia_adv_fix_mean"
    elif "mia_adv_fix_mean" in summary_df.columns and summary_df["mia_adv_fix_mean"].notna().any():
        x_col = "mia_adv_fix_mean"
    y_col = "tstr_balacc_mean" if "tstr_balacc_mean" in summary_df.columns else "tstr_acc_mean"
    if x_col and y_col in summary_df.columns:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for _, r in summary_df.iterrows():
            x = r.get(x_col)
            y = r.get(y_col)
            if pd.isna(x) or pd.isna(y):
                continue
            ax.scatter(
                x,
                y,
                color=method_color(r.get("method", "")),
                marker=dataset_marker(r.get("dataset", "")),
            )
        xlabel = "Attack advantage |AUC-0.5| (down)"
        if x_col == "privacy_adv_primary_mean":
            higher_better = bool(master_df.get("privacy_adv_primary_higher_better", pd.Series(False)).any())
            if higher_better:
                xlabel = "NN distance (norm median) (up)"
        ax.set_xlabel(xlabel)
        if y_col == "tstr_acc_mean":
            ylabel = "TSTR accuracy (higher is better)"
        else:
            ylabel = "TSTR balanced accuracy (higher is better)"
        ax.set_ylabel(ylabel)
        ax.set_title("Privacy vs Utility Frontier")
        if xlabel.startswith("Attack advantage"):
            ax.set_xlim(-0.01, 0.51)
        _write_fig(figures_dir / "fig_frontier_privacy_vs_utility.pdf")
    else:
        warnings.append("Privacy vs utility frontier skipped (missing advantage or tstr_balacc).")

    # README
    readme_path = paper_dir / "README.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_files = [p.name for p in figures_dir.glob("*.pdf")]
    table_files = [p.name for p in tables_dir.glob("*.csv")]
    readme_lines = [
        f"Timestamp: {timestamp}",
        f"Master rows: {len(master_df)}",
        f"Privacy files found: {len(privacy_files)}",
        "Privacy file list:",
    ]
    readme_lines.extend([f"- {p}" for p in privacy_used])
    readme_lines.append("Figures generated:")
    readme_lines.extend([f"- {f}" for f in fig_files])
    readme_lines.append("Tables generated:")
    readme_lines.extend([f"- {f}" for f in table_files])
    # Privacy coverage table
    coverage_rows = []
    coverage_metrics = PRIVACY_CANONICAL + [c for c in ["mia_adv_fix", "knn_mia_adv_fix"] if c in master_df.columns]
    for (ds, method), sub in master_df.groupby(["dataset", "method"], dropna=False):
        row = {"dataset": ds, "method": method}
        for c in coverage_metrics:
            row[f"{c}_non_nan"] = int(sub[c].notna().sum()) if c in sub.columns else 0
        coverage_rows.append(row)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = tables_dir / "privacy_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    readme_lines.append("Warnings:")
    if warnings:
        readme_lines.extend([f"- {w}" for w in warnings])
    else:
        readme_lines.append("- None")
    readme_lines.append("Notes:")
    readme_lines.append("- membership_audit_adv (from mia_attack.json) is a real-only membership audit; not synth-dependent.")
    readme_lines.extend(
        [
            "Reproduce:",
            "python scripts/build_master_results_and_plots.py",
        ]
    )
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    # Audits + proofs
    print(master_df[["mia_auc", "knn_mia_auc"]].notna().sum())
    eps_equal = 1e-9
    eps_report = 1e-6
    if "knn_mia_auc" in master_df.columns:
        exact_zero_knn = int((master_df["knn_mia_auc"] == 0).sum())
        exact_one_knn = int((master_df["knn_mia_auc"] == 1).sum())
        print(f"[AUDIT] exact_zero_knn={exact_zero_knn}")
        print(f"[AUDIT] exact_one_knn={exact_one_knn}")
        knn_vals = master_df["knn_mia_auc"].dropna()
        if not knn_vals.empty:
            print(
                f"[AUDIT] knn_mia_auc_stats min={knn_vals.min():.10f} "
                f"max={knn_vals.max():.10f} mean={knn_vals.mean():.10f}"
            )
            below_random = int((knn_vals < 0.51).sum())
            print(f"[AUDIT] knn_mia_auc<0.51={below_random}")
            le_eps = int((knn_vals <= eps_equal).sum())
            ge_one = int((knn_vals >= 1 - eps_equal).sum())
            eq_one = int((knn_vals == 1).sum())
            near_eps = int((knn_vals <= eps_report).sum())
            print(f"[AUDIT] knn_mia_auc<=eps_equal={le_eps} eps_equal={eps_equal}")
            print(f"[AUDIT] knn_mia_auc<=eps_report={near_eps} eps_report={eps_report}")
            print(f"[AUDIT] knn_mia_auc>=1-eps_equal={ge_one}")
            print(f"[AUDIT] knn_mia_auc==1={eq_one}")
            near_zero = master_df[master_df["knn_mia_auc"] <= eps_equal].copy()
            cols = [
                c
                for c in [
                    "dataset",
                    "method",
                    "epsilon",
                    "run_dir",
                    "syn_path",
                    "knn_mia_auc_raw",
                    "knn_mia_auc",
                    "_knn_source_file",
                    "_knn_source_col",
                ]
                if c in near_zero.columns
            ]
            if not near_zero.empty:
                print(near_zero[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.10f}"))
            # Top 10 KNN rows with provenance
            cols_top = [
                c
                for c in [
                    "dataset",
                    "method",
                    "epsilon",
                    "seed",
                    "run_dir",
                    "syn_path",
                    "knn_mia_auc",
                    "knn_mia_auc_raw",
                    "auc_flipped",
                    "_knn_source_file",
                    "_knn_source_col",
                ]
                if c in master_df.columns
            ]
            top_rows = master_df[master_df["knn_mia_auc"].notna()]
            if not top_rows.empty:
                print(top_rows[cols_top].head(10).to_string(index=False, float_format=lambda x: f"{x:.10f}"))
            if "_knn_source_file" in master_df.columns:
                counts = master_df.loc[master_df["knn_mia_auc"].notna(), "_knn_source_file"].value_counts(dropna=False)
                print(counts.to_string())
    if "mia_auc" in master_df.columns and "knn_mia_auc" in master_df.columns:
        sub_corr = master_df[["mia_auc", "knn_mia_auc"]].dropna()
        corr_val = sub_corr.corr().iloc[0, 1] if len(sub_corr) >= 2 else np.nan
        print(f"[AUDIT] mia_knn_corr={corr_val}")
        diffs = (master_df["mia_auc"] - master_df["knn_mia_auc"]).abs()
        eq_mask = diffs <= eps_equal
        near_mask = diffs <= eps_report
        eq_count = int(eq_mask.sum())
        near_count = int(near_mask.sum())
        print(f"[AUDIT] mia_knn_equal_count={eq_count} eps_equal={eps_equal}")
        print(f"[AUDIT] mia_knn_near_equal_count={near_count} eps_report={eps_report}")
        eq_rows = master_df[eq_mask]
        cols = [c for c in ["dataset", "method", "run_dir", "syn_path", "mia_auc", "knn_mia_auc"] if c in eq_rows.columns]
        if not eq_rows.empty:
            print(eq_rows[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    base_rows = len(base_df)
    final_rows = len(master_df)
    print(f"[AUDIT] base_rows={base_rows} final_master_rows={final_rows} diff={base_rows - final_rows}")
    if not knn_table.empty:
        knn_table_ok = int(knn_table["knn_mia_auc"].notna().sum())
        master_run_set = set(master_df["run_dir_key"].dropna().astype(str)) if "run_dir_key" in master_df.columns else set()
        master_syn_set = set(master_df["syn_path_key"].dropna().astype(str)) if "syn_path_key" in master_df.columns else set()
        ok_rows = knn_table[knn_table["knn_mia_auc"].notna()].copy()
        merged_by_run_dir = int(ok_rows["run_dir"].isin(master_run_set).sum()) if "run_dir" in ok_rows.columns else 0
        merged_by_syn_path = int(ok_rows["syn_path"].isin(master_syn_set).sum()) if "syn_path" in ok_rows.columns else 0
        missing_ok = ok_rows.copy()
        if "run_dir" in missing_ok.columns:
            missing_ok = missing_ok[~missing_ok["run_dir"].isin(master_run_set)]
        if "syn_path" in missing_ok.columns:
            missing_ok = missing_ok[~missing_ok["syn_path"].isin(master_syn_set)]
        knn_non_nan = int(master_df["knn_mia_auc"].notna().sum())
        mia_non_nan = int(master_df["mia_auc"].notna().sum())
        total_synth_runs = len(master_df)
        knn_cov = (knn_non_nan / total_synth_runs) if total_synth_runs else 0.0
        mia_cov = (mia_non_nan / total_synth_runs) if total_synth_runs else 0.0
        print(f"[AUDIT] total_synth_runs={total_synth_runs}")
        print(f"[AUDIT] knn_table_ok={knn_table_ok}")
        print(f"[AUDIT] knn_non_nan={knn_non_nan}")
        print(f"[AUDIT] knn_coverage_ratio={knn_cov:.3f}")
        print(f"[AUDIT] mia_non_nan={mia_non_nan}")
        print(f"[AUDIT] mia_coverage_ratio={mia_cov:.3f}")
        print(f"[AUDIT] merged_by_run_dir={merged_by_run_dir}")
        print(f"[AUDIT] merged_by_syn_path={merged_by_syn_path}")
        if not missing_ok.empty:
            cols = [c for c in ["run_dir", "syn_path", "knn_mia_auc", "_knn_source_file"] if c in missing_ok.columns]
            print(missing_ok[cols].head(20).to_string(index=False))

        # Top-10 unique by run_dir or syn_path
        knn_join = knn_table.copy()
        join_cols = [c for c in ["dataset", "method", "epsilon", "seed", "run_dir_key", "syn_path_key"] if c in master_df.columns]
        if "run_dir" in knn_join.columns and "run_dir_key" in master_df.columns:
            knn_join = knn_join.merge(
                master_df[join_cols].drop_duplicates(),
                left_on="run_dir",
                right_on="run_dir_key",
                how="left",
            )
        elif "syn_path" in knn_join.columns and "syn_path_key" in master_df.columns:
            knn_join = knn_join.merge(
                master_df[join_cols].drop_duplicates(),
                left_on="syn_path",
                right_on="syn_path_key",
                how="left",
            )
        key_col = "run_dir" if "run_dir" in knn_join.columns else "syn_path"
        knn_join = knn_join[knn_join["knn_mia_auc"].notna()].copy()
        knn_join = knn_join.drop_duplicates(subset=[key_col])
        cols = [c for c in ["dataset", "method", "epsilon", "seed", "run_dir", "syn_path", "knn_mia_auc"] if c in knn_join.columns]
        print(knn_join.sort_values("knn_mia_auc", ascending=False)[cols].head(10).to_string(index=False))
        print(knn_join.sort_values("knn_mia_auc", ascending=True)[cols].head(10).to_string(index=False))

        # Top-10 MIA rows
        if 'mia_table' in locals() and not mia_table.empty:
            mia_join = mia_table.copy()
            if "run_dir_key" in mia_join.columns and "run_dir_key" in master_df.columns:
                mia_join = mia_join.merge(
                    master_df[["dataset", "method", "epsilon", "seed", "run_dir_key", "syn_path_key"]].drop_duplicates(),
                    on="run_dir_key",
                    how="left",
                )
            key_col_m = "run_dir" if "run_dir" in mia_join.columns else "syn_path"
            if key_col_m in mia_join.columns:
                mia_join = mia_join.drop_duplicates(subset=[key_col_m])
            cols_m = [c for c in ["dataset", "method", "epsilon", "seed", "run_dir", "syn_path", "mia_auc"] if c in mia_join.columns]
            if not mia_join.empty and "mia_auc" in mia_join.columns:
                mia_join = mia_join[mia_join["mia_auc"].notna()]
                if not mia_join.empty:
                    print(mia_join.sort_values("mia_auc", ascending=False)[cols_m].head(10).to_string(index=False))

    if warnings:
        print("[WARN] Warnings:")
        for w in warnings:
            print(f" - {w}")


if __name__ == "__main__":
    main()
