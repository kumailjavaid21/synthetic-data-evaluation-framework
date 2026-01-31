import argparse
import importlib.util
import sys
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
util_path = ROOT / "src" / "utility_tstr_and_corr.py"
spec = importlib.util.spec_from_file_location("utility_tstr_and_corr", str(util_path))
util = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = util
spec.loader.exec_module(util)


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


def _normalize_dir_key(p: str) -> str:
    s = str(p).strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]:/", "", s)
    s = re.sub(r"^\./", "", s)
    for suffix in ["/synth.csv", "/synth_unrepaired.csv", "/synth_base.csv"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def _find_runs(outputs_root: Path, datasets: Iterable[str]) -> Dict[Path, Path]:
    keep = {d.upper() for d in datasets}
    by_dir: Dict[Path, Path] = {}
    for p in outputs_root.rglob("synth.csv"):
        if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
            continue
        ds = _infer_dataset_from_path(p)
        if ds and ds.upper() not in keep:
            continue
        by_dir[p.parent] = p
    for p in outputs_root.rglob("synth.npy"):
        if "paper_package" in p.parts or "FINAL_RESULTS_PACKAGE" in p.parts:
            continue
        if p.parent in by_dir:
            continue
        ds = _infer_dataset_from_path(p)
        if ds and ds.upper() not in keep:
            continue
        by_dir[p.parent] = p
    return by_dir


def _load_existing(backfill_path: Path) -> pd.DataFrame:
    if not backfill_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(backfill_path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill TSTR + CorrMAD for existing synth runs.")
    ap.add_argument("--outputs_root", type=str, default="outputs")
    ap.add_argument("--datasets", nargs="*", default=["A", "B", "C"])
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    datasets = [d.upper() for d in args.datasets]

    backfill_path = outputs_root / "tstr_corr_backfill_incremental.csv"
    existing = _load_existing(backfill_path)
    existing_keys = set(existing["run_dir_key"].dropna().astype(str)) if "run_dir_key" in existing.columns else set()

    runs = _find_runs(outputs_root, datasets)
    rows = []
    skipped = []

    for run_dir, synth_path in sorted(runs.items()):
        run_dir_key = _normalize_dir_key(str(run_dir))
        if run_dir_key in existing_keys:
            continue
        dataset = _infer_dataset_from_path(synth_path)
        if not dataset or dataset.upper() not in datasets:
            continue
        if synth_path.suffix.lower() == ".npy":
            skipped.append((run_dir, "SKIP_NPY_ONLY"))
            continue
        method = _infer_method_from_path(synth_path)
        eps_total, eps_train, eps_repair = _infer_eps_from_path(synth_path)
        seed = _infer_seed_from_path(synth_path)
        split_seed = seed if seed is not None else 0

        util.SPLIT_SEED = split_seed
        util.RANDOM_SEED = split_seed
        metrics = util.evaluate_one(dataset, synth_path, method, eps_total)
        if metrics.get("invalid_for_tstr") or metrics.get("fail_code") in {
            "FAIL_MISSING_LABEL",
            "FAIL_SYNTH_SINGLE_CLASS",
            "FAIL_REAL_SINGLE_CLASS_TEST",
            "FAIL_SINGLE_CLASS_REALTRAIN",
        }:
            skipped.append((run_dir, metrics.get("fail_code", "SKIP_INVALID_LABEL")))
            continue
        if metrics.get("status") != "OK":
            skipped.append((run_dir, metrics.get("fail_code", "SKIP_FAIL")))
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "epsilon": eps_total,
                "epsilon_train": eps_train,
                "epsilon_repair": eps_repair,
                "run_dir": str(run_dir),
                "run_dir_key": run_dir_key,
                "syn_path": str(synth_path),
                "tstr_acc": metrics.get("tstr_acc"),
                "tstr_balacc": metrics.get("tstr_balacc"),
                "tstr_f1macro": metrics.get("tstr_f1macro"),
                "pearson_mad": metrics.get("pearson_mad"),
                "corr_status": metrics.get("corr_status"),
                "corr_error": metrics.get("corr_error"),
                "status": metrics.get("status"),
                "fail_code": metrics.get("fail_code"),
                "error_msg": metrics.get("error_msg"),
                "invalid_for_tstr": metrics.get("invalid_for_tstr"),
                "_mtime": time.time(),
            }
        )

    df_new = pd.DataFrame(rows)
    if existing.empty and df_new.empty:
        print("[WARN] No backfill rows produced.")
        return

    df_all = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    if "run_dir_key" in df_all.columns:
        ok_flag = df_all.get("status", pd.Series([None] * len(df_all))).astype(str).str.upper() == "OK"
        df_all["_ok_first"] = ok_flag.astype(int)
        if "_mtime" not in df_all.columns:
            df_all["_mtime"] = 0.0
        df_all = df_all.sort_values(by=["_ok_first", "_mtime"], ascending=[False, False])
        df_all = df_all.drop_duplicates(subset=["run_dir_key"], keep="first")
        df_all = df_all.drop(columns=["_ok_first"], errors="ignore")

    backfill_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(backfill_path, index=False)
    print(f"[OK] wrote {backfill_path} rows={len(df_all)} new={len(df_new)} skipped={len(skipped)}")
    if skipped:
        counts = pd.Series([s for _, s in skipped]).value_counts()
        print("[INFO] skipped reasons:")
        for k, v in counts.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
