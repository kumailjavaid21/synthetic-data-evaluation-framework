import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_path(p: str) -> str:
    s = str(p).strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]:/", "", s)
    s = re.sub(r"^\./", "", s)
    return s


def normalize_dir_key(p: str) -> str:
    s = normalize_path(p)
    for suffix in ["/synth.csv", "/synth_unrepaired.csv", "/synth_base.csv"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def ensure_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "syn_path_key" not in df.columns:
        if "syn_path" in df.columns:
            df["syn_path_key"] = df["syn_path"].map(normalize_path)
        elif "synth_path" in df.columns:
            df["syn_path_key"] = df["synth_path"].map(normalize_path)
    if "run_dir_key" not in df.columns:
        if "run_dir" in df.columns:
            df["run_dir_key"] = df["run_dir"].map(normalize_dir_key)
        elif "run_path" in df.columns:
            df["run_dir_key"] = df["run_path"].map(normalize_dir_key)
        elif "output_dir" in df.columns:
            df["run_dir_key"] = df["output_dir"].map(normalize_dir_key)
    return df


def infer_synth_kind(path_str: str) -> str:
    p = str(path_str).lower()
    if p.endswith("synth_unrepaired.csv"):
        return "synth_unrepaired"
    if p.endswith("synth.csv"):
        return "synth"
    return "unknown"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_dist_cols(df: pd.DataFrame, prefix: str) -> list:
    cols = []
    for c in df.columns:
        c_l = c.lower()
        if prefix in c_l and "mean" in c_l:
            cols.append(c)
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="outputs/master_results.csv")
    parser.add_argument("--knn_csv", type=str, default="outputs/knn_mia_results.csv")
    parser.add_argument("--fig_dir", type=str, default="outputs/paper_package/figures")
    args = parser.parse_args()

    master_path = Path(args.master_csv)
    knn_path = Path(args.knn_csv)
    fig_dir = Path(args.fig_dir)
    ensure_dir(fig_dir)

    if not master_path.exists():
        raise FileNotFoundError(master_path)
    if not knn_path.exists():
        raise FileNotFoundError(knn_path)

    master = pd.read_csv(master_path)
    knn = pd.read_csv(knn_path)

    required_cols = [
        "knn_mia_auc_raw",
        "knn_mia_auc",
        "status",
        "fail_code",
        "k",
        "generated_at",
    ]
    missing_required = [c for c in required_cols if c not in knn.columns]
    if missing_required:
        print(f"[WARN] knn_mia_results missing columns: {missing_required}")

    knn = ensure_key_cols(knn)
    master = ensure_key_cols(master)

    knn_auc = pd.to_numeric(knn.get("knn_mia_auc"), errors="coerce")
    knn_auc_raw = pd.to_numeric(knn.get("knn_mia_auc_raw"), errors="coerce")

    exact_one = int((knn_auc == 1.0).sum())
    raw_le = int((knn_auc_raw <= 1e-6).sum())
    raw_ge = int((knn_auc_raw >= 1 - 1e-6).sum())
    print(f"[STAT] knn_mia_auc==1.0 count={exact_one}")
    print(f"[STAT] knn_mia_auc_raw<=1e-6 count={raw_le}")
    print(f"[STAT] knn_mia_auc_raw>=1-1e-6 count={raw_ge}")

    def _summary(s: pd.Series, name: str):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            print(f"[STAT] {name}: no data")
            return
        print(
            f"[STAT] {name} min={s.min():.6f} median={s.median():.6f} "
            f"mean={s.mean():.6f} max={s.max():.6f}"
        )

    _summary(knn_auc_raw, "knn_mia_auc_raw")
    _summary(knn_auc, "knn_mia_auc")

    def _uniq_counts(s: pd.Series, name: str):
        s = pd.to_numeric(s, errors="coerce").dropna()
        uniq = s.nunique()
        print(f"[STAT] {name} unique_values={uniq}")
        if not s.empty:
            print(s.value_counts().head(10).to_string())

    _uniq_counts(knn_auc_raw, "knn_mia_auc_raw")
    _uniq_counts(knn_auc, "knn_mia_auc")

    # Join coverage: syn_path_key first, then run_dir_key
    join_syn = None
    if "syn_path_key" in knn.columns and "syn_path_key" in master.columns:
        join_syn = master.merge(
            knn[["syn_path_key", "knn_mia_auc"]].dropna(),
            on="syn_path_key",
            how="left",
            suffixes=("", "_knn"),
        )
        coverage_syn = int(join_syn["knn_mia_auc_knn"].notna().sum())
        print(f"[COVER] joined_by_syn_path_key={coverage_syn}")
    else:
        print("[COVER] syn_path_key not available for join")

    if "run_dir_key" in knn.columns and "run_dir_key" in master.columns:
        join_run = master.merge(
            knn[["run_dir_key", "knn_mia_auc"]].dropna(),
            on="run_dir_key",
            how="left",
            suffixes=("", "_knn"),
        )
        coverage_run = int(join_run["knn_mia_auc_knn"].notna().sum())
        print(f"[COVER] joined_by_run_dir_key={coverage_run}")
    else:
        print("[COVER] run_dir_key not available for join")

    # Group counts for AUC==1.0 in master
    master_knn = master.copy()
    if "knn_mia_auc" in master_knn.columns:
        master_knn["knn_mia_auc"] = pd.to_numeric(master_knn["knn_mia_auc"], errors="coerce")
        master_knn = master_knn[master_knn["knn_mia_auc"] == 1.0]
        if not master_knn.empty:
            if "syn_path" in master_knn.columns:
                master_knn["synth_kind"] = master_knn["syn_path"].map(infer_synth_kind)
            group_cols = [c for c in ["dataset", "method", "epsilon", "repaired_flag", "synth_kind"] if c in master_knn.columns]
            if group_cols:
                grouped = master_knn.groupby(group_cols).size().reset_index(name="count")
                grouped = grouped.sort_values("count", ascending=False)
                print("[GROUP] top AUC==1.0 groups:")
                print(grouped.head(20).to_string(index=False))

    # Figures
    plt.rcParams.update({"font.size": 10})
    if knn_auc.notna().any():
        plt.figure(figsize=(6, 4))
        plt.hist(knn_auc.dropna(), bins=30)
        plt.xlabel("knn_mia_auc (symmetric)")
        plt.ylabel("count")
        plt.title("KNN-MIA AUC (symmetric) histogram")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_knn_auc_hist.pdf", bbox_inches="tight")
        plt.close()

    if knn_auc_raw.notna().any():
        plt.figure(figsize=(6, 4))
        plt.hist(knn_auc_raw.dropna(), bins=30)
        plt.xlabel("knn_mia_auc_raw")
        plt.ylabel("count")
        plt.title("KNN-MIA AUC raw histogram")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_knn_auc_raw_hist.pdf", bbox_inches="tight")
        plt.close()

    # Boxplot by epsilon for DP methods
    if "method" in master.columns and "epsilon" in master.columns and "knn_mia_auc" in master.columns:
        dp = master[master["method"].astype(str).str.contains("DP", case=False, na=False)].copy()
        dp["epsilon"] = pd.to_numeric(dp["epsilon"], errors="coerce")
        dp["knn_mia_auc"] = pd.to_numeric(dp["knn_mia_auc"], errors="coerce")
        dp = dp.dropna(subset=["epsilon", "knn_mia_auc"])
        if not dp.empty:
            eps_vals = sorted(dp["epsilon"].unique().tolist())
            data = [dp[dp["epsilon"] == e]["knn_mia_auc"].values for e in eps_vals]
            if data:
                plt.figure(figsize=(8, 4))
                plt.boxplot(data, tick_labels=[str(e) for e in eps_vals], showfliers=False)
                plt.xlabel("epsilon")
                plt.ylabel("knn_mia_auc")
                plt.title("KNN-MIA AUC by epsilon (DP methods)")
                plt.tight_layout()
                plt.savefig(fig_dir / "fig_knn_auc_by_epsilon_box.pdf", bbox_inches="tight")
                plt.close()

    # Distance gap scatter if available
    member_cols = find_dist_cols(knn, "member_dist")
    nonmember_cols = find_dist_cols(knn, "nonmember_dist")
    if member_cols and nonmember_cols:
        m_col = member_cols[0]
        n_col = nonmember_cols[0]
        gap = pd.to_numeric(knn[n_col], errors="coerce") - pd.to_numeric(knn[m_col], errors="coerce")
        auc_vals = pd.to_numeric(knn["knn_mia_auc"], errors="coerce")
        mask = gap.notna() & auc_vals.notna()
        if mask.any():
            plt.figure(figsize=(6, 4))
            plt.scatter(gap[mask], auc_vals[mask], s=10)
            plt.xlabel("mean_nonmember_dist - mean_member_dist")
            plt.ylabel("knn_mia_auc")
            plt.title("KNN-MIA distance gap vs AUC")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_knn_dist_gap_scatter.pdf", bbox_inches="tight")
            plt.close()

            auc1 = knn[knn["knn_mia_auc"] == 1.0]
            if m_col in auc1.columns and n_col in auc1.columns and not auc1.empty:
                mean_gap = (
                    pd.to_numeric(auc1[n_col], errors="coerce") - pd.to_numeric(auc1[m_col], errors="coerce")
                ).dropna()
                if not mean_gap.empty:
                    pos = (mean_gap > 0).mean()
                    print(f"[STAT] AUC==1 gap positive fraction={pos:.3f}")

    # Interpretation
    interpretation = []
    if exact_one > 0:
        if raw_le > 0 and raw_ge == 0:
            interpretation.append("AUC==1 mainly from raw near 0: direction-flipped perfect separation.")
        elif raw_ge > 0 and raw_le == 0:
            interpretation.append("AUC==1 mainly from raw near 1: perfect separation.")
        else:
            interpretation.append("AUC==1 appears from both raw near 0 and near 1.")

    if knn_auc.dropna().nunique() <= 10:
        interpretation.append("Warning: knn_mia_auc has low unique values (possible discretization).")

    if "method" in master.columns and "knn_mia_auc" in master.columns:
        top_groups = None
        mk = master.copy()
        mk["knn_mia_auc"] = pd.to_numeric(mk["knn_mia_auc"], errors="coerce")
        mk = mk[mk["knn_mia_auc"] == 1.0]
        if not mk.empty:
            grp_cols = [c for c in ["dataset", "method", "epsilon"] if c in mk.columns]
            if grp_cols:
                top_groups = mk.groupby(grp_cols).size().reset_index(name="count").sort_values("count", ascending=False)
        if top_groups is not None and not top_groups.empty:
            interpretation.append("AUC==1 concentrated in: " + "; ".join(
                top_groups.head(3).apply(lambda r: f"{r.get('dataset')} {r.get('method')} eps={r.get('epsilon')} ({r.get('count')})", axis=1).tolist()
            ))

    print("[INTERPRET]")
    if interpretation:
        for line in interpretation:
            print(f"- {line}")
    else:
        print("- No strong concentration detected.")

    # Extra group summaries for raw==0 and auc==1
    if "knn_mia_auc" in knn.columns:
        knn_one = knn[pd.to_numeric(knn["knn_mia_auc"], errors="coerce") == 1.0]
        if not knn_one.empty:
            grp_cols = [c for c in ["dataset", "method", "epsilon", "synth_kind", "repaired_flag"] if c in knn_one.columns]
            if grp_cols:
                print("[GROUP] knn_mia_auc==1.0 (knn results):")
                print(
                    knn_one.groupby(grp_cols).size().reset_index(name="count").sort_values("count", ascending=False).head(20).to_string(index=False)
                )
    if "knn_mia_auc_raw" in knn.columns:
        knn_raw_zero = knn[pd.to_numeric(knn["knn_mia_auc_raw"], errors="coerce") == 0.0]
        if not knn_raw_zero.empty:
            grp_cols = [c for c in ["dataset", "method", "epsilon", "synth_kind", "repaired_flag"] if c in knn_raw_zero.columns]
            if grp_cols:
                print("[GROUP] knn_mia_auc_raw==0.0 (knn results):")
                print(
                    knn_raw_zero.groupby(grp_cols).size().reset_index(name="count").sort_values("count", ascending=False).head(20).to_string(index=False)
                )

    # List generated PDFs
    pdfs = sorted(p.name for p in fig_dir.glob("fig_knn_*.pdf"))
    print("[FILES] " + ", ".join(pdfs))


if __name__ == "__main__":
    main()
