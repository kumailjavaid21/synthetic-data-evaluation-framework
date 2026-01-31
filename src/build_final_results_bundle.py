import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_privacy_agreement import run_audit as run_privacy_agreement
from scripts.make_dpcr_ablation_artifacts import make_dpcr_ablation_artifacts
from scripts.privacy_metrics_utils import (
    ensure_advantage_columns,
    choose_primary_privacy_adv,
    privacy_metric_higher_better,
)
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


def norm_path_key(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\\", "/").lower()
    while s.endswith("/"):
        s = s[:-1]
    return s


def safe_float(col):
    return pd.to_numeric(col, errors="coerce")


def pick_frontier_privacy(df: pd.DataFrame) -> str:
    if "privacy_adv_primary" in df.columns and df["privacy_adv_primary"].notna().any():
        return "privacy_adv_primary"
    if "knn_mia_adv_fix" in df.columns and df["knn_mia_adv_fix"].notna().any():
        return "knn_mia_adv_fix"
    if "mia_adv_fix" in df.columns and df["mia_adv_fix"].notna().any():
        return "mia_adv_fix"
    return ""


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "run_dir_key" not in df.columns and "run_dir" in df.columns:
        df["run_dir_key"] = df["run_dir"].map(norm_path_key)
    if "syn_path_key" not in df.columns:
        if "syn_path" in df.columns:
            df["syn_path_key"] = df["syn_path"].map(norm_path_key)
        elif "synth_path" in df.columns:
            df["syn_path_key"] = df["synth_path"].map(norm_path_key)
    return df


def ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = safe_float(df[c])
    return df


def bootstrap_mean_ci(values, n_boot=2000, ci=0.95, seed=0):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0, np.nan
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    if len(vals) < 2:
        return mean, np.nan, np.nan, int(len(vals)), std
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(vals[idx])
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return mean, lo, hi, int(len(vals)), std


def _read_json_table(path: Path) -> pd.DataFrame:
    try:
        data = path.read_text(encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    try:
        obj = pd.read_json(data)
        if isinstance(obj, pd.DataFrame):
            return obj
    except Exception:
        pass
    try:
        parsed = json.loads(data)
    except Exception:
        return pd.DataFrame()
    if isinstance(parsed, list):
        return pd.DataFrame(parsed)
    if isinstance(parsed, dict):
        if "results" in parsed and isinstance(parsed["results"], list):
            return pd.DataFrame(parsed["results"])
        # Dict keyed by run id
        rows = []
        for k, v in parsed.items():
            if isinstance(v, dict):
                r = v.copy()
                r["run_key"] = k
                rows.append(r)
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "run_dir_key" not in df.columns:
        if "run_dir" in df.columns:
            df["run_dir_key"] = df["run_dir"].map(norm_path_key)
        elif "run_path" in df.columns:
            df["run_dir_key"] = df["run_path"].map(norm_path_key)
        elif "output_dir" in df.columns:
            df["run_dir_key"] = df["output_dir"].map(norm_path_key)
    if "syn_path_key" not in df.columns:
        if "syn_path" in df.columns:
            df["syn_path_key"] = df["syn_path"].map(norm_path_key)
        elif "synth_path" in df.columns:
            df["syn_path_key"] = df["synth_path"].map(norm_path_key)
    return df


def _dedup_latest(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "generated_at" in df.columns:
        df["_ts"] = pd.to_datetime(df["generated_at"], errors="coerce")
    else:
        df["_ts"] = pd.NaT
    df = df.sort_values(by=["_ts"], ascending=True)
    return df.drop_duplicates(subset=key_cols, keep="last").drop(columns=["_ts"], errors="ignore")


def load_preferred_knn(outputs_root: Path) -> pd.DataFrame:
    path = outputs_root / "knn_mia_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_key_cols(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "OK"]
    else:
        df = df[pd.to_numeric(df.get("knn_mia_auc"), errors="coerce").notna()]
    df["knn_mia_auc"] = pd.to_numeric(df.get("knn_mia_auc"), errors="coerce")
    keep = ["run_dir_key", "syn_path_key", "knn_mia_auc", "generated_at"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = _dedup_latest(df, ["run_dir_key", "syn_path_key"])
    df["_knn_source_file"] = "outputs/knn_mia_results.csv"
    df["_knn_source_col"] = "knn_mia_auc"
    df["_mtime"] = path.stat().st_mtime
    return df[["run_dir_key", "syn_path_key", "knn_mia_auc", "_knn_source_file", "_knn_source_col", "generated_at", "_mtime"]]


def load_preferred_mia(outputs_root: Path) -> pd.DataFrame:
    path = outputs_root / "mia_results_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_key_cols(df)
    if "mia_auc" not in df.columns:
        return pd.DataFrame()
    df["mia_auc"] = pd.to_numeric(df["mia_auc"], errors="coerce")
    df = df[df["mia_auc"].notna()]
    if df.empty:
        return pd.DataFrame()
    df = _dedup_latest(df, ["run_dir_key", "syn_path_key"])
    df["_mia_source_file"] = str(path.as_posix())
    df["_mia_source_col"] = "mia_auc"
    df["_mtime"] = path.stat().st_mtime
    return df[["run_dir_key", "syn_path_key", "mia_auc", "_mia_source_file", "_mia_source_col", "_mtime"]]


def load_mia_override(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] mia_override not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_key_cols(df)
    if "mia_auc" not in df.columns:
        return pd.DataFrame()
    df["mia_auc"] = pd.to_numeric(df["mia_auc"], errors="coerce")
    df = df[df["mia_auc"].notna()]
    if df.empty:
        return pd.DataFrame()
    df = _dedup_latest(df, ["run_dir_key", "syn_path_key"])
    df["_mia_source_file"] = str(path.as_posix())
    df["_mia_source_col"] = "mia_auc"
    df["_mtime"] = path.stat().st_mtime
    return df[["run_dir_key", "syn_path_key", "mia_auc", "_mia_source_file", "_mia_source_col", "_mtime"]]


def dedup_source(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if df.empty or key_col not in df.columns:
        return df
    df = df.copy()
    df["_nnz"] = df.notna().sum(axis=1)
    if "_mtime" not in df.columns:
        df["_mtime"] = np.nan
    df = df.sort_values(by=["_nnz", "_mtime"], ascending=[False, False])
    return df.drop_duplicates(subset=[key_col], keep="first").drop(columns=["_nnz"], errors="ignore")


def summary_mean_std(df: pd.DataFrame, group_cols, metrics):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            vals = safe_float(g[m])
            row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            row[f"{m}_n"] = int(vals.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summary_mean_ci_long(df: pd.DataFrame, group_cols, metrics):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for m in metrics:
            mean, lo, hi, n, std = bootstrap_mean_ci(g[m], n_boot=2000, ci=0.95, seed=0)
            row = base.copy()
            row.update(
                {
                    "metric": m,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "std": std,
                    "n": n,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summary_mean_ci_wide(df: pd.DataFrame, group_cols, metrics):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            mean, lo, hi, n, std = bootstrap_mean_ci(g[m], n_boot=2000, ci=0.95, seed=0)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
            row[f"{m}_n"] = n
            row[f"{m}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_frontier(
    df: pd.DataFrame, utility_col: str, privacy_col: str, privacy_higher_better: bool = False
) -> pd.Series:
    df = df.copy()
    df = df[[utility_col, privacy_col]].dropna()
    if df.empty:
        return pd.Series([False] * len(df))
    idx = df.index
    util = df[utility_col].to_numpy()
    priv = df[privacy_col].to_numpy()
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        if privacy_higher_better:
            better = (util >= util[i]) & (priv >= priv[i]) & ((util > util[i]) | (priv > priv[i]))
        else:
            better = (util >= util[i]) & (priv <= priv[i]) & ((util > util[i]) | (priv < priv[i]))
        if np.any(better):
            is_pareto[i] = False
    return pd.Series(is_pareto, index=idx)


def merge_metric(df: pd.DataFrame, src: pd.DataFrame, value_col: str, source_file_col: str, source_col_col: str):
    if src.empty:
        return df
    df = df.copy()
    for col in ["run_dir_key", "syn_path_key"]:
        if col not in df.columns:
            df[col] = np.nan
    src = src.copy()
    if "run_dir_key" not in src.columns:
        src["run_dir_key"] = np.nan
    if "syn_path_key" not in src.columns:
        src["syn_path_key"] = np.nan

    by_run = src.dropna(subset=["run_dir_key"]).drop_duplicates(subset=["run_dir_key"], keep="last")
    by_syn = src.dropna(subset=["syn_path_key"]).drop_duplicates(subset=["syn_path_key"], keep="last")

    # First merge on syn_path_key
    df = df.merge(
        by_syn[["syn_path_key", value_col, source_file_col, source_col_col]],
        on="syn_path_key",
        how="left",
        suffixes=("", "_src1"),
    )
    fill_mask = df[value_col].isna() & df[f"{value_col}_src1"].notna()
    df.loc[fill_mask, value_col] = df.loc[fill_mask, f"{value_col}_src1"]
    df.loc[fill_mask, source_file_col] = df.loc[fill_mask, f"{source_file_col}_src1"]
    df.loc[fill_mask, source_col_col] = df.loc[fill_mask, f"{source_col_col}_src1"]
    df = df.drop(columns=[f"{value_col}_src1", f"{source_file_col}_src1", f"{source_col_col}_src1"], errors="ignore")

    # Then merge on run_dir_key (fill remaining only)
    df = df.merge(
        by_run[["run_dir_key", value_col, source_file_col, source_col_col]],
        on="run_dir_key",
        how="left",
        suffixes=("", "_src2"),
    )
    fill_mask = df[value_col].isna() & df[f"{value_col}_src2"].notna()
    df.loc[fill_mask, value_col] = df.loc[fill_mask, f"{value_col}_src2"]
    df.loc[fill_mask, source_file_col] = df.loc[fill_mask, f"{source_file_col}_src2"]
    df.loc[fill_mask, source_col_col] = df.loc[fill_mask, f"{source_col_col}_src2"]
    df = df.drop(columns=[f"{value_col}_src2", f"{source_file_col}_src2", f"{source_col_col}_src2"], errors="ignore")
    return df


def write_fig(path: Path):
    fig = plt.gcf()
    savefig_pdf_png(fig, path, None)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, float_format="%.12f")


def _short_method_label(name: str) -> str:
    label = str(name or "")
    if label.startswith("SDV-"):
        label = label.replace("SDV-", "", 1)
    return label.replace("GaussianCopula", "Gaussian Copula")


def plot_bars_by_dataset(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    ylabel: str,
    hline_y: float | None = None,
    y_limits: tuple[float, float] | None = None,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return False
    if summary_df[mean_col].notna().sum() == 0:
        return False

    methods = order_methods(summary_df["method"].astype(str).unique().tolist())
    datasets = order_datasets(summary_df["dataset"].astype(str).unique().tolist())
    x = np.arange(len(datasets))
    width = 0.7 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=paper_figsize(2.4))
    for m_i, m in enumerate(methods):
        sub = summary_df[summary_df["method"].astype(str) == m].set_index("dataset")
        for d_i, ds in enumerate(datasets):
            pos = x[d_i] + (m_i - (len(methods) - 1) / 2) * width
            y = sub.loc[ds, mean_col] if ds in sub.index else np.nan
            yerr = sub.loc[ds, std_col] if ds in sub.index else np.nan
            color = method_color_paper(m)
            ax.bar(
                pos,
                y,
                width=width,
                yerr=yerr,
                capsize=2.5,
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if hline_y is not None:
        ax.axhline(hline_y, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    style_axes(ax)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.22, top=0.85)
    savefig_pdf_png(fig, out_path, None)
    return True


def plot_privacy_forest_by_dataset(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    xlabel: str,
    x_limits: tuple[float, float] = (0.0, 0.5),
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return False
    if summary_df[mean_col].notna().sum() == 0:
        return False

    methods = order_methods(summary_df["method"].astype(str).unique().tolist())
    datasets = order_datasets(summary_df["dataset"].astype(str).unique().tolist())
    if not datasets:
        return False

    height = 3.6 if len(datasets) > 1 else 2.4
    fig, axes = plt.subplots(len(datasets), 1, figsize=paper_figsize(height), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = summary_df[summary_df["dataset"].astype(str) == ds].copy()
        if sub.empty:
            continue
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
            x = float(mean)
            xerr = float(std) if pd.notna(std) else 0.0
            ax.errorbar(
                x,
                i,
                xerr=xerr,
                fmt="o",
                markersize=5,
                capsize=2.5,
                elinewidth=1.0,
                color=method_color_paper(m),
            )
        ax.set_title(f"Dataset {ds}", fontsize=9, loc="left", pad=2)
        ax.axvline(0.5, linestyle="--", linewidth=0.8, alpha=0.4, color="gray")
        ax.set_xlim(*x_limits)
        style_axes(ax)

    axes[-1].set_xlabel(xlabel)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.12)
    savefig_pdf_png(fig, out_path, None)
    return True


def plot_corr_forest(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    xlabel: str,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return False
    if summary_df[mean_col].notna().sum() == 0:
        return False

    df_plot = summary_df.copy()
    df_plot[mean_col] = safe_float(df_plot[mean_col])
    df_plot[std_col] = safe_float(df_plot[std_col])
    df_plot = df_plot[df_plot[mean_col].notna()].copy()
    if df_plot.empty:
        return False

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
    return True


def frontier_ci_plot(
    summary_ci: pd.DataFrame,
    privacy_metric: str,
    utility_metric: str,
    out_path: Path,
    title: str,
    eps_df: pd.DataFrame | None = None,
    best_points: list[dict] | None = None,
    privacy_higher_better: bool = False,
):
    if summary_ci.empty or not privacy_metric:
        return False
    x_mean = f"{privacy_metric}_mean"
    x_lo = f"{privacy_metric}_ci_lo"
    x_hi = f"{privacy_metric}_ci_hi"
    y_mean = f"{utility_metric}_mean"
    y_lo = f"{utility_metric}_ci_lo"
    y_hi = f"{utility_metric}_ci_hi"
    if x_mean not in summary_ci.columns or y_mean not in summary_ci.columns:
        return False

    fig, ax = plt.subplots(figsize=paper_figsize(2.7))
    datasets = summary_ci["dataset"].dropna().astype(str).unique().tolist()
    dataset_order = order_datasets(datasets)
    for ds in dataset_order:
        sub = summary_ci[summary_ci["dataset"].astype(str) == ds]
        marker = dataset_marker(ds)
        for _, r in sub.iterrows():
            x = r.get(x_mean)
            y = r.get(y_mean)
            if pd.isna(x) or pd.isna(y):
                continue
            xerr = None
            yerr = None
            if x_lo in r and x_hi in r and pd.notna(r.get(x_lo)) and pd.notna(r.get(x_hi)):
                xerr = [[x - r.get(x_lo)], [r.get(x_hi) - x]]
            if y_lo in r and y_hi in r and pd.notna(r.get(y_lo)) and pd.notna(r.get(y_hi)):
                yerr = [[y - r.get(y_lo)], [r.get(y_hi) - y]]
            color = method_color_paper(r.get("method", ""))
            ms = 7 if "dp-diffusion+dp-cr" in str(r.get("method", "")).lower() else 5
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                marker=marker,
                linestyle="none",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=ms,
            )

    # Baseline hull per dataset (non-DP)
    base_mask = summary_ci["method"].astype(str).str.contains("sdv|ctgan|tvae|gaussiancopula", case=False, na=False)
    if base_mask.any():
        for ds in dataset_order:
            sub = summary_ci[base_mask & (summary_ci["dataset"].astype(str) == ds)]
            sub = sub[[x_mean, y_mean]].dropna().sort_values(x_mean)
            if sub.empty:
                continue
            pareto = pareto_frontier(sub, y_mean, x_mean, privacy_higher_better=privacy_higher_better)
            hull = sub.loc[pareto].sort_values(x_mean)
            if len(hull) >= 2:
                ax.plot(hull[x_mean], hull[y_mean], linestyle="-", linewidth=1, color="gray", alpha=0.5)
    else:
        print("[WARN] No baselines found for frontier hull.")

    # epsilon annotations for DP-Diffusion+DP-CR pareto points only
    if eps_df is not None and not eps_df.empty:
        for ds, g in eps_df.groupby("dataset", dropna=False):
            pareto = pareto_frontier(g, "utility_mean", "privacy_mean", privacy_higher_better=privacy_higher_better)
            pareto_idx = pareto.index[pareto]
            for _, r in g.loc[pareto_idx].iterrows():
                if pd.isna(r.get("epsilon")):
                    continue
                x = r.get("privacy_mean")
                y = r.get("utility_mean")
                if pd.isna(x) or pd.isna(y):
                    continue
                ax.text(x + 0.001, y + 0.001, f"eps={r.get('epsilon')}", fontsize=8)

    if best_points:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        dx = 0.02 * (x_max - x_min)
        dy = 0.02 * (y_max - y_min)
        for bp in best_points:
            x = bp.get("x")
            y = bp.get("y")
            label = bp.get("label")
            method = bp.get("method", "")
            ds_idx = bp.get("ds_idx", 0)
            if pd.isna(x) or pd.isna(y):
                continue
            ax.scatter(
                x,
                y,
                marker="*",
                s=140,
                color=method_color(method),
                edgecolor="black",
                linewidth=0.6,
                zorder=6,
            )
            if label:
                dy_adj = dy * (1 + 0.5 * float(ds_idx))
                text_x = min(max(x + dx, x_min), x_max)
                text_y = min(max(y + dy_adj, y_min), y_max)
                ax.text(
                    text_x,
                    text_y,
                    label,
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=0.9,
                    ),
                )

    x_vals = safe_float(summary_ci.get(x_mean))
    y_vals = safe_float(summary_ci.get(y_mean))
    x_min = float(np.nanmin(x_vals)) if x_vals.notna().any() else 0.0
    x_max = float(np.nanmax(x_vals)) if x_vals.notna().any() else 1.0
    y_min = float(np.nanmin(y_vals)) if y_vals.notna().any() else 0.0
    y_max = float(np.nanmax(y_vals)) if y_vals.notna().any() else 1.0
    ax.set_xlim(max(0.0, x_min - 0.002), x_max + 0.002)
    ax.set_ylim(y_min - 0.02, y_max + 0.02)

    if privacy_higher_better and privacy_metric in {"privacy_main", "nn_dist_norm_median"}:
        ax.set_xlabel("NN distance (norm median) (↑)")
    elif privacy_metric.endswith("_adv"):
        ax.set_xlabel("Attack advantage |AUC-0.5| (↓)")
    else:
        ax.set_xlabel("Attack AUC (↓)")
    if utility_metric == "tstr_f1macro":
        ylabel = "TSTR macro-F1 (↑)"
    else:
        ylabel = "TSTR bal. acc. (↑)"
    ax.set_ylabel(ylabel)
    style_axes(ax)

    # Inset zoom for low-AUC cluster
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        inset = inset_axes(ax, width="42%", height="42%", loc="upper left", borderpad=1.1)
        for ds in dataset_order:
            sub = summary_ci[summary_ci["dataset"].astype(str) == ds]
            marker = dataset_marker(ds)
            for _, r in sub.iterrows():
                x = r.get(x_mean)
                y = r.get(y_mean)
                if pd.isna(x) or pd.isna(y):
                    continue
                color = method_color_paper(r.get("method", ""))
                inset.scatter(x, y, marker=marker, color=color, s=12)
        low_max = 0.012
        if x_vals.notna().any():
            low_candidates = x_vals[x_vals <= 0.012]
            if not low_candidates.empty:
                low_max = float(low_candidates.max())
            low_max = max(low_max, float(x_vals.min()) + 1e-4)
        inset.set_xlim(0.0, min(0.012, low_max + 0.0005))
        inset.set_ylim(y_min - 0.02, y_max + 0.02)
        inset.tick_params(labelsize=7)
        style_axes(inset)
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6)
    except Exception:
        pass
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=method_color_paper(m),
            markeredgecolor=method_color_paper(m),
            label=m,
        )
        for m in order_methods(summary_ci["method"].astype(str).unique().tolist())
    ]
    dataset_handles = [
        Line2D([0], [0], marker=dataset_marker(ds), color="black", linestyle="none", label=f"Dataset {ds}")
        for ds in dataset_order
    ]
    ax.legend(
        handles=method_handles + dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.82)
    savefig_pdf_png(fig, out_path, None)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, default="outputs/master_results.csv")
    parser.add_argument("--out", type=str, default="final_results_bundle")
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--mia_override", type=str, default="")
    parser.add_argument("--wipe", type=int, default=1)
    args = parser.parse_args()

    if args.datasets and len(args.datasets) == 1 and "," in args.datasets[0]:
        args.datasets = [d for d in args.datasets[0].split(",") if d.strip()]

    master_path = Path(args.master)
    if not master_path.exists():
        raise FileNotFoundError(master_path)

    out_dir = Path(args.out)
    if args.wipe and out_dir.exists():
        shutil.rmtree(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    audits_dir = out_dir / "audits"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path)
    df = add_keys(df)
    if args.datasets:
        keep = [d.upper() for d in args.datasets]
        df = df[df["dataset"].astype(str).str.upper().isin(keep)].copy()
        print(f"[STAT] dataset_filter={keep} rows={len(df)}")
    if "invalid_for_tstr" in df.columns:
        invalid = df["invalid_for_tstr"]
        invalid_mask = invalid.fillna(False)
        invalid_mask = invalid_mask.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        dropped = int(invalid_mask.sum())
        if dropped:
            df = df[~invalid_mask].copy()
        print(f"[FILTER] dropped_invalid_for_tstr={dropped}")

    canonical_cols = [
        "dataset",
        "method",
        "epsilon",
        "seed",
        "repaired_flag",
        "synth_kind",
        "run_dir",
        "syn_path",
        "tstr_balacc",
        "tstr_f1macro",
        "tstr_acc",
        "pearson_mad",
        "mia_auc",
        "knn_mia_auc",
    ]
    df = ensure_cols(df, canonical_cols)
    df = ensure_numeric(
        df,
        [
            "epsilon",
            "seed",
            "tstr_balacc",
            "tstr_f1macro",
            "tstr_acc",
            "pearson_mad",
            "mia_auc",
            "knn_mia_auc",
        ],
    )

    if args.datasets and "D" in keep and "repaired_flag" in df.columns:
        method_mask = df["method"].astype(str).str.contains("dp-cr", case=False, na=False)
        repaired_num = pd.to_numeric(df["repaired_flag"], errors="coerce")
        repaired_str = df["repaired_flag"].astype(str).str.strip().str.lower()
        repaired_truthy = repaired_num.eq(1) | repaired_str.isin(["true", "1", "1.0", "yes", "y"])
        drop_mask = method_mask & repaired_truthy & (df["dataset"].astype(str).str.upper() == "D")
        dropped = int(drop_mask.sum())
        if dropped:
            df = df[~drop_mask].copy()
        print(f"[FILTER] dropped_dpcr_collapsed_rows={dropped}")

    # Clear any existing privacy values/provenance before merging canonical sources
    for col in ["mia_auc", "knn_mia_auc"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = np.nan
    for col in ["_mia_source_file", "_mia_source_col", "_knn_source_file", "_knn_source_col"]:
        df[col] = ""

    before_rows = len(df)
    outputs_root = Path(args.outputs_root)
    knn_pref = load_preferred_knn(outputs_root)
    if not knn_pref.empty:
        knn_pref = dedup_source(knn_pref, "syn_path_key")
        df = merge_metric(df, knn_pref, "knn_mia_auc", "_knn_source_file", "_knn_source_col")
    else:
        print("[WARN] Preferred KNN source missing or empty; knn_mia_auc set to NaN.")

    if args.mia_override:
        print("[WARN] mia_override ignored; hard-locked to outputs/mia_results_clean.csv.")
    mia_pref = load_preferred_mia(outputs_root)
    if not mia_pref.empty:
        mia_pref = dedup_source(mia_pref, "syn_path_key")
        df = merge_metric(df, mia_pref, "mia_auc", "_mia_source_file", "_mia_source_col")
    else:
        print("[WARN] Preferred MIA source missing or empty; mia_auc set to NaN.")

    # Standardize privacy metrics + flags
    df["mia_auc"] = safe_float(df["mia_auc"])
    df["knn_mia_auc"] = safe_float(df["knn_mia_auc"])
    df = ensure_advantage_columns(df)
    df["privacy_adv_primary"] = choose_primary_privacy_adv(df)
    df["mia_missing"] = df["mia_auc"].isna()
    df["knn_missing"] = df["knn_mia_auc"].isna()
    mia_auc_non_nan = int(pd.to_numeric(df.get("mia_auc", pd.Series(dtype=float)), errors="coerce").notna().sum())
    knn_auc_non_nan = int(pd.to_numeric(df.get("knn_mia_auc", pd.Series(dtype=float)), errors="coerce").notna().sum())
    mia_adv_non_nan = int(pd.to_numeric(df.get("mia_adv_fix", pd.Series(dtype=float)), errors="coerce").notna().sum())
    knn_adv_non_nan = int(pd.to_numeric(df.get("knn_mia_adv_fix", pd.Series(dtype=float)), errors="coerce").notna().sum())
    privacy_non_nan = int(pd.to_numeric(df.get("privacy_adv_primary", pd.Series(dtype=float)), errors="coerce").notna().sum())
    print(
        f"[STAT] mia_auc_non_nan={mia_auc_non_nan} knn_mia_auc_non_nan={knn_auc_non_nan} "
        f"mia_adv_fix_non_nan={mia_adv_non_nan} knn_mia_adv_fix_non_nan={knn_adv_non_nan} "
        f"privacy_adv_primary_non_nan={privacy_non_nan}"
    )
    priv_vals = pd.to_numeric(df.get("privacy_adv_primary", pd.Series(dtype=float)), errors="coerce")
    if priv_vals.notna().any() and float(priv_vals.median()) == 0.5:
        print("[ALERT] privacy metric saturated at 0.5; check eval leakage or parsing.")

    if len(df) != before_rows:
        print(f"[WARN] Row count changed after merge: {before_rows} -> {len(df)}; dedup by syn_path_key.")
        if "syn_path_key" in df.columns:
            df["_nnz"] = df.notna().sum(axis=1)
            df = df.sort_values(by="_nnz", ascending=False).drop_duplicates(subset=["syn_path_key"], keep="first").drop(columns=["_nnz"])

    # DP flags
    df["is_dp"] = df["method"].astype(str).str.contains("dp", case=False, na=False)
    df["is_non_dp"] = ~df["is_dp"]

    # Utility and privacy main
    df["utility_main"] = df["tstr_balacc"]
    df.loc[df["utility_main"].isna(), "utility_main"] = df["tstr_acc"]
    privacy_metric = pick_frontier_privacy(df)
    privacy_higher_better = privacy_metric_higher_better(df, privacy_metric)
    df["privacy_main"] = df[privacy_metric] if privacy_metric else np.nan

    best_points = []
    score_lambda = 50.0
    datasets_in_df = order_datasets(df["dataset"].dropna().astype(str).unique().tolist())
    for ds_idx, ds in enumerate(datasets_in_df):
        dsub = df[df["dataset"].astype(str).str.upper() == ds].copy()
        dsub = dsub[dsub["method"].astype(str).str.startswith("DP-")]
        dsub["utility_main"] = pd.to_numeric(dsub["utility_main"], errors="coerce")
        dsub["privacy_main"] = pd.to_numeric(dsub["privacy_main"], errors="coerce")
        dsub = dsub.dropna(subset=["utility_main", "privacy_main"])
        if dsub.empty:
            continue
        if privacy_higher_better:
            scores = dsub["utility_main"] + score_lambda * dsub["privacy_main"]
        else:
            scores = dsub["utility_main"] - score_lambda * dsub["privacy_main"]
        best = dsub.loc[scores.idxmax()]
        method = str(best.get("method", ""))
        if "dp-cr" in method.lower() or "dpcr" in method.lower():
            label_method = "DP-Diffusion+DP-CR"
        else:
            label_method = "DP-Diffusion"
        eps = best.get("epsilon_train")
        if pd.isna(eps):
            eps = best.get("epsilon")
        seed = best.get("seed")
        label = f"Best DP ({ds}): {label_method} eps={eps} seed={int(seed) if pd.notna(seed) else 'NA'}"
        best_points.append(
            {
                "dataset": ds,
                "x": float(best.get("privacy_main")),
                "y": float(best.get("utility_main")),
                "label": label,
                "method": method,
                "ds_idx": ds_idx,
            }
        )

    # Summary tables
    metrics = [
        "utility_main",
        "tstr_f1macro",
        "privacy_main",
        "pearson_mad",
        "mia_auc",
        "knn_mia_auc",
        "mia_adv_fix",
        "knn_mia_adv_fix",
        "privacy_adv_primary",
    ]
    summary_std = summary_mean_std(df, ["dataset", "method"], metrics)
    write_csv(summary_std, tables_dir / "summary_mean_std.csv")

    summary_ci_long = summary_mean_ci_long(df, ["dataset", "method"], metrics)
    write_csv(summary_ci_long, tables_dir / "summary_mean_ci95.csv")

    summary_ci = summary_mean_ci_wide(df, ["dataset", "method"], metrics)

    # Main table (one row per dataset)
    main_rows = []
    for ds in sorted(df["dataset"].dropna().astype(str).unique()):
        sub = summary_ci[summary_ci["dataset"].astype(str) == ds].copy()
        if sub.empty:
            main_rows.append({"dataset": ds})
            continue
        sub["utility_main_mean"] = safe_float(sub["utility_main_mean"])
        sub["tstr_acc_mean"] = safe_float(sub["tstr_acc_mean"]) if "tstr_acc_mean" in sub.columns else np.nan
        priv_mean_col = "privacy_main_mean" if "privacy_main_mean" in sub.columns else ""

        sort_cols = ["utility_main_mean", "tstr_acc_mean"]
        asc = [False, False]
        if priv_mean_col:
            sub_non_nan = sub[sub[priv_mean_col].notna()]
            if not sub_non_nan.empty:
                sub = sub_non_nan
            sort_cols.append(priv_mean_col)
            asc.append(False if privacy_higher_better else True)
        best_overall = sub.sort_values(sort_cols, ascending=asc).iloc[0]

        dp_mask = sub["method"].astype(str).str.contains("dp", case=False, na=False)
        non_dp_mask = ~dp_mask
        sub_dp = sub[dp_mask]
        sub_non_dp = sub[non_dp_mask]
        best_dp = sub_dp.sort_values(sort_cols, ascending=asc).iloc[0] if not sub_dp.empty else pd.Series()
        best_non_dp = sub_non_dp.sort_values(sort_cols, ascending=asc).iloc[0] if not sub_non_dp.empty else pd.Series()

        row = {
            "dataset": ds,
            "best_dp_method": best_dp.get("method", np.nan),
            "best_overall_method": best_overall.get("method", np.nan),
            "best_non_dp_method": best_non_dp.get("method", np.nan),
            "dp_utility_mean": best_dp.get("utility_main_mean", np.nan),
            "dp_utility_std": best_dp.get("utility_main_std", np.nan),
            "dp_utility_ci_lo": best_dp.get("utility_main_ci_lo", np.nan),
            "dp_utility_ci_hi": best_dp.get("utility_main_ci_hi", np.nan),
            "dp_privacy_mean": best_dp.get("privacy_main_mean", np.nan),
            "dp_privacy_std": best_dp.get("privacy_main_std", np.nan),
            "dp_privacy_ci_lo": best_dp.get("privacy_main_ci_lo", np.nan),
            "dp_privacy_ci_hi": best_dp.get("privacy_main_ci_hi", np.nan),
            "dp_mia_auc_mean": best_dp.get("mia_auc_mean", np.nan),
            "dp_knn_mia_auc_mean": best_dp.get("knn_mia_auc_mean", np.nan),
            "dp_mia_adv_mean_fix": best_dp.get("mia_adv_fix_mean", np.nan),
            "dp_knn_mia_adv_mean_fix": best_dp.get("knn_mia_adv_fix_mean", np.nan),
            "overall_utility_mean": best_overall.get("utility_main_mean", np.nan),
            "overall_privacy_mean": best_overall.get("privacy_main_mean", np.nan),
            "overall_mia_auc_mean": best_overall.get("mia_auc_mean", np.nan),
            "overall_knn_mia_auc_mean": best_overall.get("knn_mia_auc_mean", np.nan),
            "overall_mia_adv_mean_fix": best_overall.get("mia_adv_fix_mean", np.nan),
            "overall_knn_mia_adv_mean_fix": best_overall.get("knn_mia_adv_fix_mean", np.nan),
            "delta_dp_vs_best_non_dp_utility": (
                best_dp.get("utility_main_mean", np.nan) - best_non_dp.get("utility_main_mean", np.nan)
                if not best_dp.empty and not best_non_dp.empty
                else np.nan
            ),
            "delta_dp_vs_best_non_dp_privacy": (
                best_dp.get("privacy_main_mean", np.nan) - best_non_dp.get("privacy_main_mean", np.nan)
                if not best_dp.empty and not best_non_dp.empty
                else np.nan
            ),
            "winner_label": (
                "DP wins"
                if not best_dp.empty
                and best_dp.get("utility_main_mean", np.nan)
                >= best_overall.get("utility_main_mean", np.nan) - 1e-12
                else "Overall wins"
            ),
        }
        main_rows.append(row)

    main_df = pd.DataFrame(main_rows)
    write_csv(main_df, tables_dir / "main_table_overall.csv")

    # Winners (overall + DP)
    winners = []
    winners_dp = []
    for ds in sorted(df["dataset"].dropna().astype(str).unique()):
        sub = summary_std[summary_std["dataset"].astype(str) == ds].copy()
        if sub.empty:
            winners.append({"dataset": ds})
            winners_dp.append({"dataset": ds})
            continue
        sub["utility_main_mean"] = safe_float(sub.get("utility_main_mean"))
        sort_cols = ["utility_main_mean"]
        asc = [False]
        if "privacy_main_mean" in sub.columns:
            sub_non_nan = sub[sub["privacy_main_mean"].notna()]
            if not sub_non_nan.empty:
                sub = sub_non_nan
            sort_cols.append("privacy_main_mean")
            asc.append(False if privacy_higher_better else True)
        winners.append(sub.sort_values(sort_cols, ascending=asc).iloc[0])
        sub_dp = sub[sub["method"].astype(str).str.contains("dp", case=False, na=False)]
        winners_dp.append(sub_dp.sort_values(sort_cols, ascending=asc).iloc[0] if not sub_dp.empty else {"dataset": ds})

    # Enforce privacy coverage for winners
    winners_overall_df = pd.DataFrame(winners)
    winners_dp_df = pd.DataFrame(winners_dp)
    for c in ["privacy_main_mean", "mia_adv_fix_mean", "knn_mia_adv_fix_mean"]:
        if c not in winners_overall_df.columns:
            winners_overall_df[c] = np.nan
        if c not in winners_dp_df.columns:
            winners_dp_df[c] = np.nan
    missing_overall = winners_overall_df[winners_overall_df["privacy_main_mean"].isna()]
    missing_dp = winners_dp_df[winners_dp_df["privacy_main_mean"].isna()]
    if not missing_overall.empty or not missing_dp.empty:
        print(
            "[WARN] Winner privacy metrics missing; see final_results_bundle/audits/privacy_missing_examples.csv"
        )

    write_csv(winners_overall_df, tables_dir / "winners_overall.csv")
    write_csv(winners_dp_df, tables_dir / "winners_dp_only.csv")

    # Money tables per dataset (preserve)
    for ds in sorted(df["dataset"].dropna().astype(str).unique()):
        sub = df[df["dataset"].astype(str) == ds].copy()
        if sub.empty:
            continue
        group_cols = ["dataset", "method"]
        if "epsilon" in sub.columns and sub["epsilon"].notna().any():
            group_cols.append("epsilon")
        if "repaired_flag" in sub.columns:
            group_cols.append("repaired_flag")
        if "synth_kind" in sub.columns:
            group_cols.append("synth_kind")
        metrics = [
            "utility_main",
            "tstr_acc",
            "privacy_main",
            "mia_auc",
            "knn_mia_auc",
            "mia_adv_fix",
            "knn_mia_adv_fix",
            "pearson_mad",
        ]
        money = summary_mean_std(sub, group_cols, metrics)
        money = money.rename(
            columns={
                "utility_main_mean": "utility_mean",
                "utility_main_std": "utility_std",
                "tstr_acc_mean": "utility_acc_mean",
                "tstr_acc_std": "utility_acc_std",
                "pearson_mad_mean": "corr_mean",
                "pearson_mad_std": "corr_std",
            }
        )
        if "privacy_main_mean" in money.columns:
            money = money.rename(
                columns={
                    "privacy_main_mean": "privacy_mean",
                    "privacy_main_std": "privacy_std",
                }
            )
        seed_counts = (
            sub.groupby(group_cols, dropna=False)["seed"]
            .nunique()
            .reset_index(name="seed_count")
        )
        money = money.merge(seed_counts, on=group_cols, how="left")
        money["n_seeds"] = pd.to_numeric(money["seed_count"], errors="coerce").fillna(0).astype(int)
        n_cols = [c for c in money.columns if c.endswith("_n")]
        if n_cols:
            n_vals = money[n_cols].apply(pd.to_numeric, errors="coerce")
            max_n = n_vals.max(axis=1).fillna(money["n_seeds"])
            money["n_seeds"] = np.maximum(max_n, money["n_seeds"]).astype(int)
        money = money.drop(columns=["seed_count"], errors="ignore")
        best_util = money["utility_mean"].max()
        money["utility_winner"] = (money["utility_mean"] == best_util).astype(int)
        if "privacy_mean" in money.columns and money["privacy_mean"].notna().any():
            best_priv = money["privacy_mean"].max() if privacy_higher_better else money["privacy_mean"].min()
            money["privacy_winner"] = (money["privacy_mean"] == best_priv).astype(int)
        else:
            money["privacy_winner"] = 0
        pareto = (
            pareto_frontier(money, "utility_mean", "privacy_mean", privacy_higher_better=privacy_higher_better)
            if "privacy_mean" in money.columns
            else pd.Series([False] * len(money))
        )
        money["pareto_winner"] = pareto.astype(int).values if len(pareto) == len(money) else 0

        base_mask = ~money["method"].astype(str).str.contains("dp", case=False, na=False)
        base = money[base_mask]
        if not base.empty and "utility_mean" in base.columns:
            base_row = base.sort_values("utility_mean", ascending=False).iloc[0]
            money["delta_vs_baseline_utility"] = money["utility_mean"] - base_row.get("utility_mean")
            if "privacy_mean" in money.columns:
                money["delta_vs_baseline_privacy"] = money["privacy_mean"] - base_row.get("privacy_mean")
            else:
                money["delta_vs_baseline_privacy"] = np.nan
        else:
            money["delta_vs_baseline_utility"] = np.nan
            money["delta_vs_baseline_privacy"] = np.nan

        if money["delta_vs_baseline_utility"].notna().sum() == 0:
            raise RuntimeError(f"delta_vs_baseline_utility all NaN for dataset {ds}")
        if money["delta_vs_baseline_privacy"].notna().sum() == 0:
            print(f"[WARN] delta_vs_baseline_privacy all NaN for dataset {ds}")
        if (money["n_seeds"] < 1).any():
            raise RuntimeError(f"n_seeds < 1 detected for dataset {ds}")
        for col in money.columns:
            if col.endswith("_n"):
                too_large = money[col].notna() & (money[col].astype(int) > money["n_seeds"])
                if too_large.any():
                    raise RuntimeError(f"n_seeds mismatch in {col} for dataset {ds}")

        write_csv(money, tables_dir / f"money_table_{ds}.csv")

    # Audits
    coverage_rows = []
    for ds, g in df.groupby("dataset", dropna=False):
        row = {"dataset": ds}
        for m in ["utility_main", "tstr_f1macro", "privacy_main", "pearson_mad"]:
            row[f"{m}_non_nan"] = int(safe_float(g[m]).notna().sum())
        coverage_rows.append(row)
    overall = {"dataset": "ALL"}
    for m in ["utility_main", "tstr_f1macro", "privacy_main", "pearson_mad"]:
        overall[f"{m}_non_nan"] = int(safe_float(df[m]).notna().sum())
    coverage_rows.append(overall)
    write_csv(pd.DataFrame(coverage_rows), audits_dir / "coverage_counts.csv")

    if privacy_metric:
        priv_vals = safe_float(df[privacy_metric])
        df_priv = df.copy()
        df_priv[privacy_metric] = priv_vals
        df_priv = df_priv[df_priv[privacy_metric].notna()]
        if privacy_higher_better:
            worst = df_priv.sort_values(privacy_metric, ascending=True).head(10)
            best = df_priv.sort_values(privacy_metric, ascending=False).head(10)
        else:
            worst = df_priv.sort_values(privacy_metric, ascending=False).head(10)
            best = df_priv.sort_values(privacy_metric, ascending=True).head(10)
        extreme = pd.concat([worst, best], ignore_index=True)
        cols = [c for c in ["dataset", "method", "epsilon", "seed", "syn_path", "run_dir", privacy_metric] if c in extreme.columns]
        write_csv(extreme[cols], audits_dir / "extreme_privacy.csv")

    missing_mask = df["utility_main"].isna() | df["privacy_main"].isna()
    missing = df[missing_mask].head(50)
    cols = [c for c in ["dataset", "method", "epsilon", "seed", "syn_path", "run_dir", "utility_main", "privacy_main"] if c in missing.columns]
    write_csv(missing[cols], audits_dir / "missing_metrics_samples.csv")

    # Privacy consistency audit
    eps_equal = 1e-9
    joint = df[df["mia_auc"].notna() & df["knn_mia_auc"].notna()].copy()
    if not joint.empty:
        mia = safe_float(joint["mia_auc"]).to_numpy()
        knn = safe_float(joint["knn_mia_auc"]).to_numpy()
        pearson = float(np.corrcoef(mia, knn)[0, 1]) if len(mia) > 1 else np.nan
        mia_rank = pd.Series(mia).rank().to_numpy()
        knn_rank = pd.Series(knn).rank().to_numpy()
        spearman = float(np.corrcoef(mia_rank, knn_rank)[0, 1]) if len(mia_rank) > 1 else np.nan
        abs_diff = np.abs(mia - knn)
        equal_count = int(np.sum(abs_diff <= eps_equal))
        equal_fraction = float(equal_count / len(joint)) if len(joint) else np.nan
        audit_row = {
            "mia_non_nan": int(df["mia_auc"].notna().sum()),
            "knn_non_nan": int(df["knn_mia_auc"].notna().sum()),
            "both_non_nan": int(len(joint)),
            "mia_knn_equal_count": equal_count,
            "mia_knn_equal_fraction": equal_fraction,
            "eps_equal": eps_equal,
            "mia_knn_pearson": pearson,
            "mia_knn_spearman": spearman,
            "mia_min": float(np.nanmin(mia)),
            "mia_median": float(np.nanmedian(mia)),
            "mia_mean": float(np.nanmean(mia)),
            "mia_max": float(np.nanmax(mia)),
            "knn_min": float(np.nanmin(knn)),
            "knn_median": float(np.nanmedian(knn)),
            "knn_mean": float(np.nanmean(knn)),
            "knn_max": float(np.nanmax(knn)),
        }
        summary_df = pd.DataFrame([audit_row])
        samples = joint.copy()
        samples["abs_diff"] = abs_diff
        top20 = samples.sort_values("abs_diff", ascending=True).head(20)[
            ["dataset", "method", "epsilon", "seed", "mia_auc", "knn_mia_auc", "abs_diff"]
        ]
        top20["record_type"] = "sample"
        summary_df["record_type"] = "summary"
        out_audit = pd.concat([summary_df, top20], ignore_index=True)
        write_csv(out_audit, audits_dir / "privacy_consistency.csv")
    else:
        write_csv(
            pd.DataFrame([{"record_type": "summary", "both_non_nan": 0, "eps_equal": eps_equal}]),
            audits_dir / "privacy_consistency.csv",
        )

    # Privacy provenance audit
    mia_non_nan = int(df["mia_auc"].notna().sum())
    knn_non_nan = int(df["knn_mia_auc"].notna().sum())
    both_non_nan = int(joint.shape[0]) if not joint.empty else 0
    equal_count = int(np.sum(np.abs(safe_float(joint["mia_auc"]) - safe_float(joint["knn_mia_auc"])) < 1e-9)) if both_non_nan else 0
    equal_fraction = float(equal_count / both_non_nan) if both_non_nan else np.nan

    if both_non_nan:
        prov = joint.copy()
        prov["abs_diff"] = np.abs(safe_float(prov["mia_auc"]) - safe_float(prov["knn_mia_auc"]))
        prov["is_equal"] = prov["abs_diff"] <= eps_equal
        group_cols = ["_mia_source_file", "_mia_source_col", "_knn_source_file", "_knn_source_col"]
        grouped = (
            prov.groupby(group_cols, dropna=False)
            .agg(n_rows=("is_equal", "size"), n_equal=("is_equal", "sum"))
            .reset_index()
        )
        grouped["equal_fraction"] = grouped["n_equal"] / grouped["n_rows"]
        grouped["mia_non_nan"] = mia_non_nan
        grouped["knn_non_nan"] = knn_non_nan
        grouped["both_non_nan"] = both_non_nan
        grouped["equal_count"] = equal_count
        grouped["equal_fraction_total"] = equal_fraction
        grouped["eps_equal"] = eps_equal
        write_csv(grouped, audits_dir / "privacy_provenance_audit.csv")

        top_equal = prov[prov["is_equal"]].copy()
        sort_cols = [c for c in ["dataset", "method", "epsilon"] if c in top_equal.columns]
        if sort_cols:
            top_equal = top_equal.sort_values(sort_cols)
        cols = [
            c
            for c in [
                "dataset",
                "method",
                "epsilon",
                "seed",
                "run_dir",
                "syn_path",
                "mia_auc",
                "knn_mia_auc",
                "_mia_source_file",
                "_mia_source_col",
                "_knn_source_file",
                "_knn_source_col",
            ]
            if c in top_equal.columns
        ]
        write_csv(top_equal[cols].head(30), audits_dir / "privacy_equal_rows_top30.csv")
    else:
        write_csv(
            pd.DataFrame(
                [
                    {
                        "mia_non_nan": mia_non_nan,
                        "knn_non_nan": knn_non_nan,
                        "both_non_nan": both_non_nan,
                        "equal_count": equal_count,
                        "equal_fraction_total": equal_fraction,
                    }
                ]
            ),
            audits_dir / "privacy_provenance_audit.csv",
        )
        write_csv(pd.DataFrame(), audits_dir / "privacy_equal_rows_top30.csv")

    print(f"[AUDIT] mia_non_nan={mia_non_nan}")
    print(f"[AUDIT] knn_non_nan={knn_non_nan}")
    print(f"[AUDIT] both_non_nan={both_non_nan}")
    print(f"[AUDIT] mia_knn_equal_count={equal_count}")
    print(f"[AUDIT] mia_knn_equal_fraction={equal_fraction}")
    if both_non_nan:
        top_groups = grouped.sort_values("equal_fraction", ascending=False).head(5)
        print(top_groups.to_string(index=False))
    else:
        print("[AUDIT] provenance_groups=0")

    # DP-CR effect summary (paired deltas)
    dpcr_repaired = df[df["method"].astype(str).str.contains("dp-diffusion\\+dp-cr", case=False, na=False)].copy()
    if "synth_kind" in dpcr_repaired.columns:
        dpcr_unrepaired = dpcr_repaired[dpcr_repaired["synth_kind"].astype(str).str.contains("unrepaired", case=False, na=False)].copy()
        dpcr_repaired = dpcr_repaired[~dpcr_repaired["synth_kind"].astype(str).str.contains("unrepaired", case=False, na=False)].copy()
    else:
        dpcr_unrepaired = dpcr_repaired[dpcr_repaired.get("repaired_flag") == False].copy()
        dpcr_repaired = dpcr_repaired[dpcr_repaired.get("repaired_flag") != False].copy()

    pair_key = ["dataset", "seed"]
    if "epsilon_train" in df.columns and "epsilon_repair" in df.columns:
        pair_key += ["epsilon_train", "epsilon_repair"]
    elif "epsilon" in df.columns:
        pair_key += ["epsilon"]
    if "run_dir_key" in df.columns:
        pair_key += ["run_dir_key"]

    for c in pair_key:
        if c not in dpcr_repaired.columns:
            dpcr_repaired[c] = np.nan
        if c not in dpcr_unrepaired.columns:
            dpcr_unrepaired[c] = np.nan

    pairs = dpcr_repaired.merge(
        dpcr_unrepaired,
        on=pair_key,
        suffixes=("_rep", "_unrep"),
        how="inner",
    )

    dpcr_rows = []
    if not pairs.empty:
        pairs["delta_utility"] = pairs["utility_main_rep"] - pairs["utility_main_unrep"]
        pairs["delta_privacy"] = pairs["privacy_main_rep"] - pairs["privacy_main_unrep"]
        pairs["delta_corr"] = pairs["pearson_mad_rep"] - pairs["pearson_mad_unrep"]
        pairs["delta_f1"] = pairs["tstr_f1macro_rep"] - pairs["tstr_f1macro_unrep"]

        for ds, g in pairs.groupby("dataset", dropna=False):
            util_mean, util_lo, util_hi, n_util, _ = bootstrap_mean_ci(g["delta_utility"], n_boot=2000, ci=0.95, seed=0)
            priv_mean, priv_lo, priv_hi, n_priv, _ = bootstrap_mean_ci(g["delta_privacy"], n_boot=2000, ci=0.95, seed=0)
            dpcr_rows.append(
                {
                    "dataset": ds,
                    "n_pairs": int(len(g)),
                    "delta_utility_mean": util_mean,
                    "delta_utility_ci_lo": util_lo,
                    "delta_utility_ci_hi": util_hi,
                    "delta_privacy_mean": priv_mean,
                    "delta_privacy_ci_lo": priv_lo,
                    "delta_privacy_ci_hi": priv_hi,
                    "delta_corr_mean": float(safe_float(g["delta_corr"]).mean()),
                    "delta_f1_mean": float(safe_float(g["delta_f1"]).mean()),
                }
            )
    write_csv(pd.DataFrame(dpcr_rows), tables_dir / "dpcr_effect_summary.csv")

    # Figures
    set_paper_style()
    summary_for_plot = summary_mean_std(
        df,
        ["dataset", "method"],
        ["utility_main", "tstr_acc", "mia_adv_fix", "knn_mia_adv_fix", "pearson_mad"],
    )
    plot_bars_by_dataset(
        summary_for_plot,
        "utility_main",
        figures_dir / "fig_utility_tstr_balacc_bar.pdf",
        "Utility (mean +/- std)",
        "TSTR balanced accuracy (higher is better)",
    )
    plot_bars_by_dataset(
        summary_for_plot,
        "tstr_acc",
        figures_dir / "fig_utility_tstr_acc_bar.pdf",
        "TSTR accuracy (mean +/- std)",
        "TSTR accuracy (higher is better)",
    )
    if "pearson_mad_mean" in summary_for_plot.columns and summary_for_plot["pearson_mad_mean"].notna().any():
        plot_corr_forest(
            summary_for_plot,
            "pearson_mad",
            figures_dir / "fig_corr_pearson_mad_bar.pdf",
            "Pearson MAD (down)",
        )
    else:
        print("[WARN] pearson_mad missing or empty; skipping fig_corr_pearson_mad_bar.pdf")
    if "mia_adv_fix_mean" in summary_for_plot.columns and summary_for_plot["mia_adv_fix_mean"].notna().any():
        plot_privacy_forest_by_dataset(
            summary_for_plot,
            "mia_adv_fix",
            figures_dir / "fig_privacy_mia_adv_bar.pdf",
            "Attack advantage |AUC-0.5| (down)",
            x_limits=(0.0, 0.5),
        )
        src = figures_dir / "fig_privacy_mia_adv_bar.pdf"
        dst = figures_dir / "fig_privacy_mia_auc_bar.pdf"
        if src.exists():
            shutil.copy2(src, dst)
    else:
        print("[WARN] mia_adv missing or empty; skipping fig_privacy_mia_adv_bar.pdf")

    if "knn_mia_adv_fix_mean" in summary_for_plot.columns and summary_for_plot["knn_mia_adv_fix_mean"].notna().any():
        plot_privacy_forest_by_dataset(
            summary_for_plot,
            "knn_mia_adv_fix",
            figures_dir / "fig_privacy_knn_mia_adv_bar.pdf",
            "Attack advantage |AUC-0.5| (down)",
            x_limits=(0.0, 0.5),
        )
        src = figures_dir / "fig_privacy_knn_mia_adv_bar.pdf"
        dst = figures_dir / "fig_privacy_knn_mia_auc_bar.pdf"
        if src.exists():
            shutil.copy2(src, dst)
    else:
        print("[WARN] knn_mia_adv missing or empty; skipping fig_privacy_knn_mia_adv_bar.pdf")

    eps_df_balacc = pd.DataFrame()
    eps_df_f1 = pd.DataFrame()
    if privacy_metric:
        dpcr = df[df["method"].astype(str).str.contains("dp-diffusion\+dp-cr", case=False, na=False)].copy()
        if not dpcr.empty and dpcr["epsilon"].notna().any():
            grp = dpcr.groupby(["dataset", "method", "epsilon"], dropna=False)
            eps_df_balacc = grp.agg(utility_mean=("utility_main", "mean"), privacy_mean=(privacy_metric, "mean")).reset_index()
            eps_df_f1 = grp.agg(utility_mean=("tstr_f1macro", "mean"), privacy_mean=(privacy_metric, "mean")).reset_index()

        frontier_ci_plot(
            summary_ci,
            "privacy_main",
            "utility_main",
            figures_dir / "fig_frontier_privacy_vs_utility_ci95.pdf",
            "Privacy-Utility Frontier (mean +/- 95% CI)",
            eps_df=eps_df_balacc,
            best_points=best_points,
            privacy_higher_better=privacy_higher_better,
        )
        frontier_ci_plot(
            summary_ci,
            "privacy_main",
            "tstr_f1macro",
            figures_dir / "fig_frontier_privacy_vs_f1macro_ci95.pdf",
            "Privacy-Utility Frontier (F1 macro, mean +/- 95% CI)",
            eps_df=eps_df_f1,
            privacy_higher_better=privacy_higher_better,
        )
    else:
        print("[WARN] No privacy metric available for frontier plots.")

    # README
    readme = out_dir / "README.txt"
    lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Source: outputs/master_results.csv (canonical merged results).",
        "Tables:",
        "- main_table_overall.csv: best DP vs best overall and deltas vs best non-DP.",
        "- summary_mean_ci95.csv: bootstrap 95% CI per dataset+method (long format).",
        "- summary_mean_std.csv: mean/std per dataset+method.",
        "- dpcr_effect_summary.csv: DP-CR paired deltas with CI.",
        "- money_table_<DS>.csv, winners_overall.csv, winners_dp_only.csv.",
        "Figures:",
        "- utility/privacy/correlation bar charts (mean +/- std).",
        "- frontier plots with mean +/- 95% CI and baseline hull.",
        "Primary privacy indicator is log10 of normalized NN distance median (higher is safer). Raw nn_dist_norm_median is retained in artifacts."
        if privacy_higher_better
        else "Privacy advantage: lower is better (|AUC-0.5| in [0, 0.5]).",
        "Utility higher is better. CI is bootstrap over seeds.",
        "Note: membership_audit_adv (from mia_attack.json) is real-only and not synth-dependent.",
        "Floating-point equality: comparisons use tolerance (1e-9) to avoid rounding artifacts.",
        "Apparent equality at 6 decimals can occur even when exact metrics differ.",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    # DP-CR ablation artifacts
    make_dpcr_ablation_artifacts(
        input_auto=True,
        out_tables_dir=tables_dir,
        out_figures_dir=figures_dir,
    )
    dpcr_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_dpcr_ablation_tables.py"),
        "--master",
        str(master_path),
        "--out_dir",
        str(tables_dir),
        "--fig_dir",
        str(figures_dir),
    ]
    subprocess.run(dpcr_cmd, check=True)
    dpcr_pub_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_dpcr_ablation_publication_tables.py"),
        "--bundle_dir",
        str(out_dir),
        "--out_dir",
        str(tables_dir),
    ]
    subprocess.run(dpcr_pub_cmd, check=True)

    # Privacy agreement audit
    run_privacy_agreement(master_path, out_dir, eps_equal=1e-9, eps_near=1e-6)

    # Console summary
    mia_non_nan = int(safe_float(df["mia_auc"]).notna().sum())
    knn_non_nan = int(safe_float(df["knn_mia_auc"]).notna().sum())
    table_files = sorted(p.name for p in tables_dir.glob("*.csv"))
    figure_files = sorted(p.name for p in figures_dir.glob("*.pdf"))
    audit_files = sorted(p.name for p in audits_dir.glob("*.csv"))
    equal_fraction = np.nan
    if (audits_dir / "privacy_consistency.csv").exists():
        pc = pd.read_csv(audits_dir / "privacy_consistency.csv")
        if "mia_knn_equal_fraction" in pc.columns:
            equal_fraction = pc["mia_knn_equal_fraction"].iloc[0]
    print(f"[OK] Bundle: {out_dir.resolve()}")
    privacy_missing = int(df["privacy_main"].isna().sum()) if "privacy_main" in df.columns else 0
    print(f"[STAT] frontier_privacy_metric={privacy_metric if privacy_metric else 'none'}")
    print(f"[STAT] mia_auc_non_nan={mia_non_nan}")
    print(f"[STAT] knn_mia_auc_non_nan={knn_non_nan}")
    mia_invalid = int(df["mia_auc_invalid"].sum()) if "mia_auc_invalid" in df.columns else 0
    knn_invalid = int(df["knn_mia_auc_invalid"].sum()) if "knn_mia_auc_invalid" in df.columns else 0
    print(f"[STAT] mia_auc_invalid={mia_invalid}")
    print(f"[STAT] knn_mia_auc_invalid={knn_invalid}")
    print(f"[STAT] privacy_missing_rows={privacy_missing}")
    print(f"[STAT] mia_knn_equal_fraction={equal_fraction}")
    print(f"[FILES] tables={len(table_files)} figures={len(figure_files)} audits={len(audit_files)}")
    print("[FILES] table_list=" + ", ".join(table_files))
    print("[FILES] figure_list=" + ", ".join(figure_files))
    print("[FILES] audit_list=" + ", ".join(audit_files))

    if args.datasets and len(args.datasets) == 1:
        ds = args.datasets[0].upper()
        mini_dir = Path("outputs") / f"paper_artifacts_{ds}"
        mini_dir.mkdir(parents=True, exist_ok=True)
        for p in figures_dir.glob("*.pdf"):
            shutil.copy2(p, mini_dir / p.name)
        for p in tables_dir.glob("*.csv"):
            shutil.copy2(p, mini_dir / p.name)
        readme = [
            f"Dataset: {ds}",
            f"Bundle: {out_dir.resolve()}",
            "Contents: figures (*.pdf) and tables (*.csv) for this dataset only.",
        ]
        (mini_dir / "README.txt").write_text("\n".join(readme), encoding="utf-8")
        print(f"[OK] Mini paper artifacts written to {mini_dir}")


if __name__ == "__main__":
    main()
