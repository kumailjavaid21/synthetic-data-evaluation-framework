"""
run_evaluate_final_selected.py

Evaluate base-paper metrics for all runs listed in outputs/final_selected_runs_manifest.csv.

Usage (PowerShell):
  python run_evaluate_final_selected.py
  python run_evaluate_final_selected.py --manifest_csv outputs/final_selected_runs_manifest.csv --label_idx -1
"""

import argparse
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval_base_paper_metrics import eval_run, QI_DEFAULT, LABEL_DEFAULT_IDX


def evaluate_row(row, label_idx):
    run_dir = Path(row.output_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    qis = QI_DEFAULT.get(row.dataset, [])
    res, _ = eval_run(
        dataset=row.dataset,
        run_dir=run_dir,
        qis=qis,
        label_idx=label_idx,
    )
    res.update(
        {
            "epsilon_total": float(row.epsilon_total),
            "epsilon_train": float(row.epsilon_train),
            "epsilon_repair": float(row.epsilon_repair),
            "seed": int(row.seed),
            "status": row.status,
        }
    )
    return res


def summarize(per_run_df):
    group_cols = ["dataset", "epsilon_total", "epsilon_train", "epsilon_repair"]
    numeric_cols = [
        c
        for c in per_run_df.columns
        if pd.api.types.is_numeric_dtype(per_run_df[c]) and c not in group_cols
    ]
    agg_map = {c: ["mean", "std"] for c in numeric_cols}
    summary = per_run_df.groupby(group_cols).agg(agg_map)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.reset_index()
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate base-paper metrics for all selected runs.")
    ap.add_argument("--manifest_csv", type=Path, default=Path("outputs/final_selected_runs_manifest.csv"))
    ap.add_argument("--out_per_run", type=Path, default=Path("outputs/basepaper_eval_per_run.csv"))
    ap.add_argument("--out_summary", type=Path, default=Path("outputs/basepaper_eval_summary.csv"))
    ap.add_argument("--label_idx", type=int, default=LABEL_DEFAULT_IDX)
    ap.add_argument("--max_workers", type=int, default=1)
    args = ap.parse_args()

    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_csv}")
    manifest = pd.read_csv(args.manifest_csv)
    required_cols = ["dataset", "epsilon_total", "epsilon_train", "epsilon_repair", "seed", "output_dir", "status"]
    for col in required_cols:
        if col not in manifest.columns:
            raise ValueError(f"Manifest missing column: {col}")

    rows = [r for r in manifest.itertuples(index=False)]
    if not rows:
        print("No rows in manifest to evaluate.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(evaluate_row, r, args.label_idx): r for r in rows if getattr(r, "status", "") != "failed"}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"[OK] Evaluated {row.dataset} eps_total={row.epsilon_total} seed={row.seed}")
            except Exception as e:
                print(f"[WARN] Failed eval for {row.dataset} eps_total={row.epsilon_total} seed={row.seed}: {e}")

    if not results:
        print("No evaluations completed.")
        return

    per_run_df = pd.DataFrame(results)
    for col in ["epsilon_total", "epsilon_train", "epsilon_repair"]:
        if col in per_run_df.columns:
            per_run_df[col] = pd.to_numeric(per_run_df[col], errors="coerce")

    args.out_per_run.parent.mkdir(parents=True, exist_ok=True)
    per_run_df.to_csv(args.out_per_run, index=False)
    print(f"[OK] Per-run base-paper metrics -> {args.out_per_run} ({len(per_run_df)} rows)")

    summary_df = summarize(per_run_df)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"[OK] Summary -> {args.out_summary} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
