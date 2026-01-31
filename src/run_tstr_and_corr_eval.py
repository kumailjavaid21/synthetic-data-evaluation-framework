"""
run_tstr_and_corr_eval.py

Batch-evaluate TSTR (Train Synthetic, Test Real) + correlation fidelity for synthetic datasets.

Usage:
  python run_tstr_and_corr_eval.py --dataset A
  python run_tstr_and_corr_eval.py --dataset B --synthetic_glob "outputs/B/**/synth*.csv"
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import src.utility_tstr_and_corr as util
from src.utility_tstr_and_corr import (
    SyntheticCandidate,
    evaluate_one,
    set_seeds,
    TARGET_MAP,
    rtr_eval,
    SPLIT_SEED,
    resolve_label_col,
    _clean_labels,
    eval_tstr_catboost_gpu,
    eval_rtr_catboost_gpu,
)


METHOD_MAP = {
    "dp_diffusion": "DP-Diffusion",
    "dp-diffusion": "DP-Diffusion",
    "dp_diffusion_dpcr": "DP-Diffusion+DP-CR",
    "dp-diffusion+dp-cr": "DP-Diffusion+DP-CR",
    "dp_diffusion_classcond": "DP-Diffusion (ClassCond)",
    "dp_diffusion_classcond_dpcr": "DP-Diffusion+DP-CR (ClassCond)",
    "dp_ctgan": "DP-CTGAN",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "gaussian_copula": "Gaussian Copula",
    "gaussian copula": "Gaussian Copula",
    "diffusion": "Tabular Diffusion",
}

def _normalize_path(p: str) -> str:
    s = str(p).strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]:/", "", s)
    s = re.sub(r"^\./", "", s)
    return s


def _normalize_dir_key(p: str) -> str:
    s = _normalize_path(p)
    for suffix in ["/synth.csv", "/synth_unrepaired.csv", "/synth_base.csv"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def map_method(name: str) -> str:
    key = name.strip().lower()
    if key in METHOD_MAP:
        return METHOD_MAP[key]
    for k, v in METHOD_MAP.items():
        if key.startswith(k):
            return v
    return name


def _infer_method_from_path(path: Path) -> str:
    p = str(path).lower()
    if "\\baselines_sdv\\ctgan\\" in p:
        return "SDV-CTGAN"
    if "\\baselines_sdv\\tvae\\" in p:
        return "SDV-TVAE"
    if "\\baselines_sdv\\gaussiancopula\\" in p:
        return "SDV-GaussianCopula"
    if "\\dp_diffusion_classcond_dpcr\\" in p and path.name.lower() == "synth_unrepaired.csv":
        return "DP-Diffusion (ClassCond, unrepaired)"
    if "\\dp_diffusion_classcond_dpcr\\" in p and path.name.lower() == "synth.csv":
        return "DP-Diffusion+DP-CR (ClassCond)"
    if "\\dp_diffusion_classcond\\" in p:
        return "DP-Diffusion (ClassCond)"
    if "\\dp_diffusion_dpcr\\" in p and path.name.lower() == "synth_unrepaired.csv":
        return "DP-Diffusion (unrepaired)"
    if "\\dp_diffusion_dpcr\\" in p and path.name.lower() == "synth.csv":
        return "DP-Diffusion+DP-CR"
    if "\\dp_diffusion\\" in p:
        return "DP-Diffusion"
    return path.parent.name


def _parse_seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.lower().startswith("seed"):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except Exception:
                    continue
    return -1


def _parse_seeds(values: Optional[List[str]]) -> List[int]:
    if not values:
        return [42, 43, 44]
    if len(values) == 1 and isinstance(values[0], str) and "," in values[0]:
        parts = [p for p in values[0].split(",") if p.strip()]
    else:
        parts = values
    out = []
    for v in parts:
        s = str(v).strip()
        if not s:
            continue
        out.append(int(s))
    return out if out else [42, 43, 44]


def _path_has_seed(path: Path, seeds: List[int]) -> bool:
    if not seeds:
        return True
    p = str(path).lower()
    return any(f"seed{seed}".lower() in p for seed in seeds)


def discover_candidates(
    dataset: str,
    synthetic_glob: Optional[str],
    include_baselines: bool,
    seeds: Optional[List[int]] = None,
    outputs_root: Optional[Path] = None,
) -> List[SyntheticCandidate]:
    if synthetic_glob:
        paths = list(Path().glob(synthetic_glob))
        out = []
        for p in paths:
            if not p.is_file():
                continue
            if seeds is not None and not _path_has_seed(p, seeds):
                continue
            out.append(SyntheticCandidate(dataset=dataset, method=_infer_method_from_path(p), epsilon=None, path=p))
        return out

    ds = dataset.upper()
    base = (outputs_root or Path("outputs")) / ds
    if not base.exists():
        return []

    patterns = []
    patterns.extend((base / "dp_diffusion").rglob("synth.csv"))
    patterns.extend((base / "dp_diffusion_dpcr").rglob("synth.csv"))
    patterns.extend((base / "dp_diffusion_classcond").rglob("synth.csv"))
    patterns.extend((base / "dp_diffusion_classcond_dpcr").rglob("synth.csv"))
    if include_baselines:
        patterns.extend((base / "baselines_sdv").rglob("synth.csv"))
    # Include unrepaired only under dp_diffusion_dpcr
    patterns.extend((base / "dp_diffusion_dpcr").rglob("synth_unrepaired.csv"))
    patterns.extend((base / "dp_diffusion_classcond_dpcr").rglob("synth_unrepaired.csv"))

    seen = set()
    candidates = []
    for p in patterns:
        if not p.is_file():
            continue
        if seeds is not None and not _path_has_seed(p, seeds):
            continue
        if p in seen:
            continue
        seen.add(p)
        method = _infer_method_from_path(p)
        candidates.append(SyntheticCandidate(dataset=ds, method=method, epsilon=None, path=p))
    return candidates


def parse_epsilon(row: Dict) -> Optional[float]:
    for key in ["epsilon", "epsilon_total", "eps_total", "epsilon_train"]:
        if key in row and pd.notna(row[key]):
            try:
                return round(float(row[key]), 3)
            except Exception:
                continue
    path = str(row.get("syn_path", row.get("path", "")))
    patterns = [r"eps[_-]?([0-9p.]+)", r"train([0-9p.]+)", r"epsilon[_-]?([0-9p.]+)"]
    for pat in patterns:
        m = re.search(pat, path, flags=re.IGNORECASE)
        if m:
            val = m.group(1).replace("p", ".")
            try:
                return round(float(val), 3)
            except Exception:
                continue
    return None


def _fail_metrics(dataset: str, method: str, epsilon: Optional[float], syn_path: Path, code: str, msg: str, label_col: Optional[str] = None, syn_label_counts: Optional[Dict] = None) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "method": map_method(method),
        "epsilon": epsilon,
        "syn_path": str(syn_path),
        "syn_path_key": _normalize_path(str(syn_path)),
        "run_dir_key": _normalize_dir_key(str(syn_path)),
        "status": "FAIL",
        "fail_code": code,
        "error_msg": msg[:200],
        "label_col": label_col,
        "syn_label_counts": syn_label_counts if syn_label_counts is not None else {},
    }


def _counts(arr: pd.Series) -> Dict[str, int]:
    vals, cnts = np.unique(arr, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnts)}


def _enforce_tstr_nan_gate(metrics: Dict[str, object]) -> Dict[str, object]:
    if metrics.get("status") != "OK":
        return metrics
    task = metrics.get("tstr_task_type")
    if task == "regression":
        tstr_cols = ["tstr_r2", "tstr_rmse"]
    else:
        tstr_cols = ["tstr_acc", "tstr_balacc", "tstr_f1macro"]
    has_nan = any(pd.isna(metrics.get(c)) for c in tstr_cols if c in metrics)
    if has_nan:
        metrics["status"] = "FAIL"
        metrics["fail_code"] = metrics.get("fail_code") or "FAIL_TSTR_NAN"
        prev = metrics.get("error_msg", "")
        metrics["error_msg"] = (prev + "|TSTR metrics contain NaN").strip("|")
    return metrics


def write_master(df_results: pd.DataFrame, master_path: Path):
    df_merge = df_results.copy()
    df_merge["method"] = df_merge["method"].apply(map_method)
    df_merge["epsilon"] = df_merge.apply(parse_epsilon, axis=1)
    df_merge["epsilon"] = pd.to_numeric(df_merge["epsilon"], errors="coerce").round(3)
    df_merge.to_csv(master_path, index=False)
    print(f"[OK] Wrote master table: {master_path}")


def rebuild_master_from_files(datasets: List[str], out_root: Path, master_path: Path):
    frames = []
    for ds in datasets:
        path = out_root / f"tstr_corr_results_{ds}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        print("[WARN] No per-dataset result files found for master rebuild.")
        return
    combined = pd.concat(frames, ignore_index=True)
    write_master(combined, master_path)


def summarize_results(df: pd.DataFrame):
    summary = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        # pick metric based on task: if any regression metrics present use r2 else acc
        if sub["rtr_r2"].notna().any():
            rtr_best = sub["rtr_r2"].dropna().max()
            tstr_best = sub["tstr_r2"].dropna().max()
            gap = rtr_best - tstr_best if pd.notna(rtr_best) and pd.notna(tstr_best) else np.nan
            summary.append((ds, rtr_best, tstr_best, gap, "r2"))
        else:
            rtr_best = sub["rtr_acc"].dropna().max()
            tstr_best = sub["tstr_acc"].dropna().max()
            gap = rtr_best - tstr_best if pd.notna(rtr_best) and pd.notna(tstr_best) else np.nan
            summary.append((ds, rtr_best, tstr_best, gap, "acc"))
    print("\n=== RTR vs TSTR summary ===")
    for ds, rtr, tstr, gap, metric in summary:
        print(f"{ds}: RTR ({metric})={rtr:.3f} | best TSTR={tstr:.3f} | gap={gap:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--synthetic_glob", type=str, default=None, help="Optional glob to override candidate discovery")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to store per-dataset results")
    parser.add_argument("--outputs_root", type=str, default="outputs", help="Root directory to discover synth outputs")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional manifest to evaluate only listed runs")
    parser.add_argument("--overwrite_results", action="store_true", default=True, help="Overwrite per-dataset result CSVs")
    parser.add_argument("--no-overwrite_results", dest="overwrite_results", action="store_false", help="Append to existing per-dataset CSVs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-dataset outputs (results + summary)")
    parser.add_argument("--rewrite_master", action="store_true", help="Rebuild master CSV from scratch (otherwise append-only)")
    parser.add_argument("--reset_master", action="store_true", help="Rebuild master CSV from scratch (overwrite file)")
    parser.add_argument("--rebuild-master", dest="rebuild_master", action="store_true", help="Rebuild master CSV from scratch (overwrite file)")
    parser.add_argument("--rebuild_master", dest="rebuild_master", action="store_true", help="Rebuild master CSV from scratch (overwrite file)")
    parser.add_argument("--include_baselines", action="store_true", default=True, help="Include baselines_sdv runs")
    parser.add_argument("--no_include_baselines", action="store_false", dest="include_baselines", help="Exclude baselines_sdv runs")
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44"], help="Seed list or comma-separated")
    parser.add_argument("--include_methods", nargs="+", default=None, help="Optional method allowlist")
    args = parser.parse_args()

    set_seeds()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    include_methods = [m.lower() for m in args.include_methods] if args.include_methods else None

    combined_results = []
    manifest_rows = None
    if args.manifest_csv:
        try:
            manifest_rows = pd.read_csv(args.manifest_csv)
        except Exception as e:
            print(f"[WARN] Could not read manifest {args.manifest_csv}: {e}")
            manifest_rows = None

    outputs_root = Path(args.outputs_root)
    util.SPLIT_DIR = outputs_root / "splits"
    for ds in args.datasets:
        ds_records = []
        ds_errors = []
        if manifest_rows is not None:
            rows_ds = manifest_rows[manifest_rows["dataset"].astype(str).str.upper() == ds.upper()]
            candidates = []
            for _, r in rows_ds.iterrows():
                out_dir = Path(r["output_dir"])
                syn_path = out_dir / "synth.csv"
                if not syn_path.exists():
                    syn_path = out_dir / "synth_base.csv"
                if not syn_path.exists():
                    print(f"[WARN] Missing synth file for {out_dir}")
                    continue
                if seeds and not _path_has_seed(syn_path, seeds):
                    continue
                cand_method = _infer_method_from_path(syn_path)
                candidates.append(SyntheticCandidate(dataset=ds, method=cand_method, epsilon=r.get("epsilon_total"), path=syn_path))
        else:
            candidates = discover_candidates(
                ds,
                args.synthetic_glob,
                args.include_baselines,
                seeds=seeds,
                outputs_root=outputs_root,
            )
            if include_methods:
                filtered = []
                for c in candidates:
                    method_label = map_method(c.method).lower()
                    if method_label in include_methods or c.method.lower() in include_methods:
                        filtered.append(c)
                candidates = filtered
        if include_methods and manifest_rows is not None:
            filtered = []
            for c in candidates:
                method_label = map_method(c.method).lower()
                if method_label in include_methods or c.method.lower() in include_methods:
                    filtered.append(c)
            candidates = filtered

        if not candidates:
            print(f"[WARN] No synthetic candidates found for dataset {ds}")
            continue
        for cand in candidates:
            try:
                # Preflight: ensure label exists and synthetic has >=2 classes before training classifier
                syn_df = pd.read_csv(cand.path)
                label_col, y_series, _ = resolve_label_col(cand.dataset, syn_df)
                if label_col is None or y_series is None:
                    cols_preview = ", ".join(list(syn_df.columns[:30]))
                    msg = f"target_not_detected_in_synthetic (dataset={cand.dataset}, cols={cols_preview})"
                    metrics = _fail_metrics(cand.dataset, cand.method, cand.epsilon, cand.path, "FAIL_MISSING_LABEL", msg)
                else:
                    cleaned = _clean_labels(y_series)
                if label_col is not None and y_series is not None and pd.Series(cleaned).nunique() < 2:
                    counts = _counts(pd.Series(cleaned))
                    msg = f"synthetic labels have <2 classes (counts={counts})"
                    metrics = _fail_metrics(
                        cand.dataset,
                        cand.method,
                        cand.epsilon,
                        cand.path,
                        "FAIL_SINGLE_CLASS",
                        msg,
                        label_col=label_col,
                        syn_label_counts=counts,
                    )
                else:
                    metrics = evaluate_one(cand.dataset, cand.path, cand.method, cand.epsilon)
                metrics = _enforce_tstr_nan_gate(metrics)
                metrics["syn_path_key"] = _normalize_path(str(cand.path))
                metrics["run_dir_key"] = _normalize_dir_key(str(cand.path))
                metrics["seed"] = _parse_seed_from_path(cand.path)
                target_col = metrics.get("target_col")
                feature_cols = metrics.get("feature_cols", [])
                if target_col and feature_cols:
                    try:
                        train_real, test_real = util.get_or_create_split(cand.dataset, feature_cols, target_col, split_seed=SPLIT_SEED)
                        rtr_metrics = rtr_eval(train_real, test_real, cand.dataset)
                        metrics.update(rtr_metrics)
                        if cand.dataset.upper() in {"A", "B", "C"}:
                            syn_df = syn_df[feature_cols + [target_col]]
                            train_real = train_real[feature_cols + [target_col]]
                            test_real = test_real[feature_cols + [target_col]]
                            metrics["real_train_label_counts"] = _counts(train_real[target_col])
                            metrics["real_test_label_counts"] = _counts(test_real[target_col])
                            metrics["syn_label_counts"] = _counts(syn_df[target_col])
                            print(
                                f"[AUDIT] {cand.dataset.upper()} label counts train={metrics['real_train_label_counts']} "
                                f"test={metrics['real_test_label_counts']} synth={metrics['syn_label_counts']}"
                            )
                            try:
                                tstr_cat = eval_tstr_catboost_gpu(train_real, test_real, syn_df, target_col)
                                rtr_cat = eval_rtr_catboost_gpu(train_real, test_real, target_col)
                            except Exception as e:
                                metrics.setdefault("notes", "")
                                metrics["notes"] = (metrics["notes"] + f"|catboost_failed:{e}").strip("|")
                                tstr_cat = {}
                                rtr_cat = {}
                            metrics.update(tstr_cat)
                            metrics.update(rtr_cat)
                            if "tstr_balacc_cat" in tstr_cat:
                                metrics["tstr_balacc"] = tstr_cat["tstr_balacc_cat"]
                            print(
                                f"[AUDIT] {cand.dataset.upper()} TSTR(catboost) balacc={tstr_cat.get('tstr_balacc_cat')} "
                                f"f1={tstr_cat.get('tstr_f1macro_cat')}"
                            )
                            print(
                                f"[AUDIT] {cand.dataset.upper()} RTR(catboost) balacc={rtr_cat.get('rtr_balacc_cat')} "
                                f"f1={rtr_cat.get('rtr_f1macro_cat')}"
                            )
                    except Exception as e:
                        metrics.setdefault("notes", "")
                        metrics["notes"] = (metrics["notes"] + f"|rtr_failed:{e}").strip("|")
                metrics["method"] = map_method(cand.method)
                metrics["epsilon"] = parse_epsilon(metrics)
                ds_records.append(metrics)
                status = metrics.get("status", "OK")
                fail_code = metrics.get("fail_code", "")
                err = metrics.get("error_msg", "")
                if status == "OK":
                    print(
                        f"[OK] {cand.path} | tstr_acc={metrics.get('tstr_acc')} "
                        f"bal_acc={metrics.get('tstr_balacc')} f1={metrics.get('tstr_f1macro')} "
                        f"rtr_acc={metrics.get('rtr_acc')} corr={metrics.get('pearson_mad')} "
                        f"label_col={metrics.get('label_col')}"
                    )
                else:
                    print(
                        f"[FAIL:{fail_code or 'UNKNOWN'}] {cand.path} :: {err} | "
                        f"label_col={metrics.get('label_col')} "
                        f"syn_counts={metrics.get('syn_label_counts')} "
                        f"real_test_counts={metrics.get('real_test_label_counts')}"
                    )
            except Exception as e:
                print(f"[WARN] Failed to evaluate {cand.path}: {e}")
                ds_errors.append({"dataset": cand.dataset, "path": str(cand.path), "error": str(e)})

        if not ds_records:
            print(f"[WARN] No evaluations completed for dataset {ds}.")
            continue

        df = pd.DataFrame(ds_records)
        for col in [
            "tstr_task_type",
            "tstr_auc",
            "tstr_acc",
            "tstr_balacc",
            "tstr_f1macro",
            "tstr_r2",
            "tstr_rmse",
            "tstr_error",
            "seed",
            "rtr_task_type",
            "rtr_auc",
            "rtr_acc",
            "rtr_balacc",
            "rtr_f1macro",
            "rtr_r2",
            "rtr_rmse",
            "tstr_balacc_cat",
            "tstr_f1macro_cat",
            "rtr_balacc_cat",
            "rtr_f1macro_cat",
            "pearson_fro",
            "pearson_mad",
            "spearman_fro",
            "spearman_mad",
            "n_num_cols_used",
            "corr_cols_used",
            "corr_rows_used",
            "corr_status",
            "corr_error",
            "heatmap_prefix",
            "status",
            "fail_code",
            "error_msg",
            "label_col",
            "syn_label_counts",
            "real_train_label_counts",
            "real_test_label_counts",
            "syn_path_key",
            "run_dir_key",
            "notes",
        ]:
            if col not in df.columns:
                df[col] = np.nan

        out_csv_path = out_root / f"tstr_corr_results_{ds}.csv"
        if not (args.overwrite_results or args.overwrite) and out_csv_path.exists():
            try:
                existing = pd.read_csv(out_csv_path)
                df_out = pd.concat([existing, df], ignore_index=True)
            except Exception as e:
                print(f"[WARN] Could not read existing {out_csv_path}: {e}. Overwriting.")
                df_out = df
        else:
            df_out = df
        df_out.to_csv(out_csv_path, index=False)
        print(f"[OK] Saved results to {out_csv_path} ({len(df_out)} rows)")

        # Dataset summary
        ok_rows = (df_out["status"] == "OK").sum() if "status" in df_out.columns else 0
        fail_rows = len(df_out) - ok_rows
        sdv_rows = df_out["method"].astype(str).str.startswith("SDV-").sum() if "method" in df_out.columns else 0
        print(f"[SUMMARY] {ds}: total={len(df_out)} OK={ok_rows} FAIL={fail_rows}")
        print(f"[SUMMARY] {ds}: SDV rows={sdv_rows}")
        if "fail_code" in df_out.columns:
            print(df_out["fail_code"].value_counts(dropna=False))

        summary_path = out_root / f"tstr_corr_summary_{ds}.json"
        with open(summary_path, "w") as f:
            json.dump({"dataset": ds, "results": df_out.to_dict("records"), "errors": ds_errors}, f, indent=2)
        print(f"[OK] Saved JSON summary to {summary_path}")

        combined_results.append(df_out)

    if not combined_results:
        print("[WARN] No evaluations completed across datasets.")
        return

    combined_df = pd.concat(combined_results, ignore_index=True)
    master_path = out_root / "tstr_corr_master.csv"
    if args.rewrite_master or args.reset_master or getattr(args, "rebuild_master", False):
        rebuild_master_from_files(args.datasets, out_root, master_path)
    else:
        if master_path.exists():
            combined_df.to_csv(master_path, mode="a", index=False, header=False)
        else:
            combined_df.to_csv(master_path, index=False)
        print(f"[OK] Appended to master table: {master_path}")

    # Summary print
    summarize_results(combined_df)


if __name__ == "__main__":
    main()
