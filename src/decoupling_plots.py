"""
decoupling_plots.py

Matplotlib-only plotting helpers for URR and privacy-efficiency visuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['lines.linewidth'] = 1.5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def _prep_fig_out(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_urr(urr_df: pd.DataFrame, dataset: str, out_path: Path):
    df = urr_df[urr_df["dataset"] == dataset]
    if df.empty:
        return
    _prep_fig_out(out_path)
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    plt.plot(df["epsilon_total"], df["URR"], marker="o", label="URR")
    if "URR_std" in df.columns and df["URR_std"].notna().any():
        y = df["URR"]
        yerr = df["URR_std"].fillna(0)
        plt.fill_between(df["epsilon_total"], y - yerr, y + yerr, color="blue", alpha=0.2, label="std")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_ylim(0.95, 1.05)
    plt.xlabel("Epsilon Total")
    plt.ylabel("Utility Recovery Ratio (URR)")
    plt.title(f"URR vs Epsilon ({dataset})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"[OK] URR plot written to {out_path.with_suffix('.pdf')}")


def plot_utility_curve(grid_df: pd.DataFrame, dataset: str, out_path: Path, util_col: str):
    df = grid_df[grid_df["dataset"] == dataset]
    if df.empty:
        return
    _prep_fig_out(out_path)
    plt.figure(figsize=(5, 4))
    df_dp = df[df.apply(lambda r: ("dp" in str(r.get("method", "")).lower()) and ("cr" not in str(r.get("method", "")).lower()), axis=1)]
    df_dpcr = df[df["method"].str.contains("cr", case=False, na=False) | df["method"].str.contains("repair", case=False, na=False)]
    plt.plot(df_dp["epsilon_total"], df_dp[util_col], marker="o", label="DP-Diffusion")
    plt.plot(df_dpcr["epsilon_total"], df_dpcr[util_col], marker="s", label="DP-CR")
    plt.xlabel("Epsilon Total")
    plt.ylabel("Utility")
    plt.title(f"Utility vs Epsilon ({dataset})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"[OK] Utility curve written to {out_path.with_suffix('.pdf')}")


def plot_epsilon_saving(eps_df: pd.DataFrame, dataset: str, out_path: Path):
    df = eps_df[eps_df["dataset"] == dataset]
    if df.empty:
        return
    _prep_fig_out(out_path)
    plt.figure(figsize=(5, 4))
    plt.plot(df["target_utility"], df["epsilon_saving"], marker="o", label="epsilon saving")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Target Utility")
    plt.ylabel("Epsilon Saving (DP - DP-CR)")
    plt.title(f"Privacy Efficiency ({dataset})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"[OK] Epsilon saving plot written to {out_path.with_suffix('.pdf')}")


def plot_hero_B(grid_df: pd.DataFrame, urr_df: pd.DataFrame, util_col: str, out_path: Path):
    df_b = grid_df[grid_df["dataset"] == "B"]
    urr_b = urr_df[urr_df["dataset"] == "B"]
    if df_b.empty or urr_b.empty:
        return
    _prep_fig_out(out_path)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df_dp = df_b[df_b.apply(lambda r: ("dp" in str(r.get("method", "")).lower()) and ("cr" not in str(r.get("method", "")).lower()), axis=1)]
    df_dpcr = df_b[df_b["method"].str.contains("cr", case=False, na=False) | df_b["method"].str.contains("repair", case=False, na=False)]
    plt.plot(df_dp["epsilon_total"], df_dp[util_col], marker="o", label="DP-Diffusion")
    plt.plot(df_dpcr["epsilon_total"], df_dpcr[util_col], marker="s", label="DP-CR")
    plt.xlabel("Epsilon Total")
    plt.ylabel("Utility")
    plt.title("B: Utility vs Epsilon")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(urr_b["epsilon_total"], urr_b["URR"], marker="o", color="green", label="URR")
    if "URR_std" in urr_b.columns and urr_b["URR_std"].notna().any():
        y = urr_b["URR"]
        yerr = urr_b["URR_std"].fillna(0)
        plt.fill_between(urr_b["epsilon_total"], y - yerr, y + yerr, color="green", alpha=0.2, label="std")
    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_ylim(0.95, 1.05)
    plt.xlabel("Epsilon Total")
    plt.ylabel("Utility Recovery Ratio (URR)")
    plt.title("B: URR")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"[OK] Hero plot written to {out_path.with_suffix('.pdf')}")
