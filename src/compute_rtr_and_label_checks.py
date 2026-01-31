import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utility_tstr_and_corr import (
    LABEL_COL,
    TARGET_CANDIDATES,
    TARGET_MAP,
    _clean_labels,
    preprocess_for_models,
    tstr_logreg,
    load_real_split,
)


ROOT = Path(".")
SEARCH_DIRS = [Path("data"), Path("datasets"), Path("inputs"), Path("outputs"), Path("frozen_splits"), Path("raw_data")]
DEFAULT_TABLE_DIR_CANDIDATES = [
    Path("submission_pack_FINAL_ABC_PATCHED") / "tables",
    Path("submission_pack_FINAL_ABC") / "tables",
    Path("publication_pack_ABC") / "tables",
    Path("final_results_bundle_ABC") / "tables",
    Path("final_results_bundle") / "tables",
    Path("package_tables_final_ABC"),
]


def _is_id_like(name: str) -> bool:
    name = name.lower()
    return bool(re.search(r"(?:^|_)(id|uid|uuid|index|user|row)(?:$|_)", name))


def infer_label_col(df: pd.DataFrame, dataset_id: str) -> Optional[str]:
    dataset_id = dataset_id.upper()
    mapped = LABEL_COL.get(dataset_id) or TARGET_MAP.get(dataset_id)
    if mapped and mapped in df.columns:
        return mapped
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand

    n = len(df)
    best = None
    best_u = None
    best_binary = False

    for col in df.columns:
        if _is_id_like(col):
            continue
        series = df[col]
        uniq = series.dropna().unique()
        u = len(uniq)
        if u <= 1:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            if pd.api.types.is_float_dtype(series):
                if u > max(20, int(0.2 * n)):
                    continue
            else:
                if u > max(50, int(0.5 * n)):
                    continue
        else:
            if u > max(50, int(0.5 * n)):
                continue

        is_binary = u == 2
        if best is None:
            best = col
            best_u = u
            best_binary = is_binary
            continue
        if is_binary and not best_binary:
            best = col
            best_u = u
            best_binary = True
            continue
        if is_binary == best_binary and best_u is not None and u < best_u:
            best = col
            best_u = u
            best_binary = is_binary

    if best is not None:
        return best
    return df.columns[-1] if len(df.columns) else None


def find_candidate_real_files(dataset_id: str) -> List[Path]:
    dataset_id = dataset_id.upper()
    name_hints = {
        "A": ["studentsperformance", "student", "A"],
        "B": ["studentinfo", "student", "B"],
        "C": ["student-mat", "student", "C"],
    }.get(dataset_id, [dataset_id])
    keywords = ["real", "train", "test", "split"]
    matches = []
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            low = path.name.lower()
            if not (low.endswith(".csv") or low.endswith(".npy")):
                continue
            path_low = str(path).lower()
            if any(h.lower() in path_low for h in name_hints) and any(k in low for k in keywords):
                matches.append(path)
    return matches


def load_real_train_test(dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_guess = LABEL_COL.get(dataset_id.upper(), "target")
    try:
        return load_real_split(dataset_id, feature_cols=[], target_col=label_guess)
    except Exception as e:
        print(f"[WARN] load_real_split failed for {dataset_id}: {e}")

    # Fallback: processed CSVs
    processed_dir = Path("data") / "processed" / dataset_id.upper()
    train_p = processed_dir / "train.csv"
    test_p = processed_dir / "test.csv"
    if train_p.exists() and test_p.exists():
        return pd.read_csv(train_p), pd.read_csv(test_p)

    # Fallback: any discovered CSVs with train/test in filename
    candidates = find_candidate_real_files(dataset_id)
    train = next((p for p in candidates if "train" in p.name.lower()), None)
    test = next((p for p in candidates if "test" in p.name.lower()), None)
    if train and test:
        return pd.read_csv(train), pd.read_csv(test)

    raise FileNotFoundError(f"No real train/test data found for dataset {dataset_id}")


def compute_label_balance(dataset_id: str, label_col: str, full_df: pd.DataFrame) -> List[Dict]:
    y = full_df[label_col]
    counts = y.value_counts(dropna=False)
    n_total = int(counts.sum())
    n_classes = len(counts)
    majority_class = str(counts.idxmax()) if n_classes else ""
    majority_acc = float(counts.max() / n_total) if n_total else math.nan
    majority_bal_acc = float(1.0 / n_classes) if n_classes else math.nan

    rows = []
    for cls, cnt in counts.items():
        rows.append(
            {
                "dataset": dataset_id,
                "label_col": label_col,
                "n_total": n_total,
                "class": str(cls),
                "count": int(cnt),
                "proportion": float(cnt / n_total) if n_total else math.nan,
                "majority_class": majority_class,
                "majority_acc": majority_acc,
                "majority_bal_acc": majority_bal_acc,
            }
        )
    return rows


def best_rtr_model(rtr_rows: List[Dict]) -> Dict:
    ok = [r for r in rtr_rows if not pd.isna(r.get("bal_acc"))]
    if not ok:
        return rtr_rows[0]
    return max(ok, key=lambda r: r.get("bal_acc", float("-inf")))


def resolve_synth_path(s: str) -> Optional[Path]:
    if not s or pd.isna(s):
        return None
    path = Path(str(s))
    if path.exists():
        return path
    alt = ROOT / path
    if alt.exists():
        return alt
    # Try search by filename under outputs/
    if path.name:
        hits = list(Path("outputs").rglob(path.name))
        if hits:
            return hits[0]
    return None


def load_synth_df(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path), "ok"
        except Exception as e:
            return None, f"read_error:{type(e).__name__}"
    if path.suffix.lower() == ".npy":
        return None, "unsupported_npy"
    return None, "unsupported_format"


def extract_label_from_synth(df: pd.DataFrame, label_col: str) -> Tuple[Optional[pd.Series], Optional[str], List[str]]:
    if label_col in df.columns:
        return df[label_col], label_col, []
    prefix = f"{label_col}_"
    prefix2 = f"{label_col}__"
    oh_cols = [c for c in df.columns if c.startswith(prefix) or c.startswith(prefix2)]
    if len(oh_cols) > 1:
        sub = df[oh_cols].to_numpy()
        idx = sub.argmax(axis=1)
        return pd.Series(idx, index=df.index, name=label_col), label_col, oh_cols
    return None, None, []


def build_paper_table_with_rtr(
    tables_dir: Path,
    out_dir: Path,
    rtr_best: Dict[str, Dict],
    datasets: List[str],
) -> Path:
    frames = []
    for ds in datasets:
        table_path = tables_dir / f"money_table_{ds}.csv"
        df = pd.read_csv(table_path)
        rtr = rtr_best.get(ds)
        rtr_row = {c: pd.NA for c in df.columns}
        rtr_row["dataset"] = ds
        rtr_row["method"] = "Real->Real (RTR)"
        if rtr:
            rtr_row["utility_mean"] = rtr.get("bal_acc")
            rtr_row["utility_acc_mean"] = rtr.get("acc")
            rtr_row["utility_main_n"] = rtr.get("n_test")
            rtr_row["tstr_acc_n"] = rtr.get("n_test")
            rtr_row["n_seeds"] = 1
        df_out = pd.concat([pd.DataFrame([rtr_row]), df], ignore_index=True)
        frames.append(df_out)
    out_path = out_dir / "paper_table_ABC_with_RTR.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def find_tables_dir(tables_dir_arg: Optional[str]) -> Path:
    if tables_dir_arg:
        p = Path(tables_dir_arg)
        if p.exists():
            return p
    for cand in DEFAULT_TABLE_DIR_CANDIDATES:
        if (cand / "money_table_A.csv").exists() and (cand / "money_table_B.csv").exists() and (cand / "money_table_C.csv").exists():
            return cand

    # Fallback: search for any dir containing money_table_A/B/C and pick most recent
    parents: Dict[Path, set] = {}
    for path in ROOT.rglob("money_table_*.csv"):
        parents.setdefault(path.parent, set()).add(path.name)
    candidates = [p for p, names in parents.items() if {"money_table_A.csv", "money_table_B.csv", "money_table_C.csv"}.issubset(names)]
    if not candidates:
        raise FileNotFoundError("Unable to locate money_table_A/B/C.csv in the repo")
    candidates.sort(key=lambda p: (p / "money_table_A.csv").stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/paper_checks")
    parser.add_argument("--tables_dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.upper() for d in args.datasets]

    rtr_rows = []
    label_balance_rows = []
    rtr_best: Dict[str, Dict] = {}
    real_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    label_cache: Dict[str, str] = {}

    for ds in datasets:
        print(f"[INFO] Processing dataset {ds}")
        candidates = find_candidate_real_files(ds)
        if candidates:
            print(f"[INFO] Candidate real files for {ds}: {', '.join(str(p) for p in candidates[:5])}")
        train_df, test_df = load_real_train_test(ds)
        real_cache[ds] = (train_df, test_df)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        label_col = infer_label_col(full_df, ds)
        if label_col is None or label_col not in full_df.columns:
            raise ValueError(f"Unable to infer label column for dataset {ds}")
        print(f"[INFO] {ds} label_col={label_col}")
        label_cache[ds] = label_col

        # Compute label balance + majority baselines
        label_balance_rows.extend(compute_label_balance(ds, label_col, full_df))

        # RTR: LogReg using existing TSTR preprocessing
        X_train, X_test, y_train, y_test, _, target_col = preprocess_for_models(train_df, test_df, dataset_id=ds)
        if target_col != label_col:
            print(f"[WARN] {ds} preprocess label_col={target_col} differs from inferred {label_col}")
        logreg_metrics = tstr_logreg(X_train, y_train, X_test, y_test)
        rtr_rows.append(
            {
                "dataset": ds,
                "label_col": target_col,
                "model": "LogisticRegression",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "acc": logreg_metrics.get("tstr_acc"),
                "bal_acc": logreg_metrics.get("tstr_balacc"),
            }
        )

        # RTR: RandomForest on same preprocessed features
        rf_row = {
            "dataset": ds,
            "label_col": target_col,
            "model": "RandomForestClassifier",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "acc": math.nan,
            "bal_acc": math.nan,
        }
        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=123)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_row["acc"] = float(accuracy_score(y_test, y_pred))
            rf_row["bal_acc"] = float(balanced_accuracy_score(y_test, y_pred))
        except Exception as e:
            print(f"[WARN] RF failed for {ds}: {type(e).__name__}: {e}")
        rtr_rows.append(rf_row)

        rtr_best[ds] = best_rtr_model([r for r in rtr_rows if r["dataset"] == ds])

    # Label sanity checks for synth outputs
    tables_dir = find_tables_dir(args.tables_dir)
    print(f"[INFO] Using tables_dir={tables_dir}")
    sanity_rows = []
    for ds in datasets:
        table_path = tables_dir / f"money_table_{ds}.csv"
        money_df = pd.read_csv(table_path)
        manifest_path = Path("outputs") / f"tstr_corr_results_{ds}.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        label_col = label_cache.get(ds) or infer_label_col(
            pd.concat([real_cache[ds][0], real_cache[ds][1]], ignore_index=True),
            ds,
        )
        for _, row in money_df.iterrows():
            method = str(row.get("method", ""))
            epsilon = row.get("epsilon")
            repaired_flag = row.get("repaired_flag")
            synth_kind = row.get("synth_kind")
            config = f"eps={epsilon}, repaired={repaired_flag}, synth_kind={synth_kind}"

            syn_path = None
            if not manifest.empty:
                sub = manifest[manifest["method"].astype(str) == method]
                if not pd.isna(epsilon):
                    sub = sub[np.isclose(pd.to_numeric(sub.get("epsilon"), errors="coerce"), float(epsilon), atol=1e-6, equal_nan=True)]
                if "repaired_flag" in sub.columns and not pd.isna(repaired_flag):
                    sub = sub[sub["repaired_flag"].astype(str) == str(repaired_flag)]
                if sub.empty:
                    sub = manifest[manifest["method"].astype(str) == method]
                if not sub.empty:
                    syn_path = sub.iloc[0].get("syn_path") or sub.iloc[0].get("synth_path")

            resolved = resolve_synth_path(syn_path) if syn_path else None
            if resolved is None:
                sanity_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "config": config,
                        "label_col": label_col,
                        "status": "missing_artifact",
                        "n_synth": math.nan,
                        "n_unique_labels": math.nan,
                        "class_counts": "",
                        "class_proportions": "",
                    }
                )
                continue

            synth_df, status = load_synth_df(resolved)
            if synth_df is None:
                sanity_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "config": config,
                        "label_col": label_col,
                        "status": status,
                        "n_synth": math.nan,
                        "n_unique_labels": math.nan,
                        "class_counts": "",
                        "class_proportions": "",
                    }
                )
                continue

            label_series, resolved_label, _ = extract_label_from_synth(synth_df, label_col)
            if label_series is None:
                sanity_rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "config": config,
                        "label_col": label_col,
                        "status": "missing_label",
                        "n_synth": int(len(synth_df)),
                        "n_unique_labels": math.nan,
                        "class_counts": "",
                        "class_proportions": "",
                    }
                )
                continue
            label_series = pd.Series(_clean_labels(label_series))
            counts = label_series.value_counts(dropna=False)
            n_synth = int(len(label_series))
            proportions = (counts / n_synth).to_dict()
            sanity_rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "config": config,
                    "label_col": resolved_label,
                    "status": "ok",
                    "n_synth": n_synth,
                    "n_unique_labels": int(len(counts)),
                    "class_counts": json.dumps({str(k): int(v) for k, v in counts.to_dict().items()}),
                    "class_proportions": json.dumps({str(k): float(v) for k, v in proportions.items()}),
                }
            )

    rtr_df = pd.DataFrame(rtr_rows)
    label_df = pd.DataFrame(label_balance_rows)
    sanity_df = pd.DataFrame(sanity_rows)

    rtr_path = out_dir / "rtr_baselines.csv"
    label_path = out_dir / "label_balance.csv"
    sanity_path = out_dir / "label_sanity_checks.csv"
    rtr_df.to_csv(rtr_path, index=False)
    label_df.to_csv(label_path, index=False)
    sanity_df.to_csv(sanity_path, index=False)

    paper_path = build_paper_table_with_rtr(tables_dir, out_dir, rtr_best, datasets)

    # Summary
    summary = []
    for ds in datasets:
        best = rtr_best.get(ds, {})
        bal = best.get("bal_acc")
        if pd.isna(bal):
            msg = f"{ds}: RTR bal_acc missing"
        else:
            msg = f"{ds}: RTR bal_acc={bal:.3f}"
            if ds in {"A", "B"} and abs(bal - 0.5) <= 0.02:
                msg += " (near chance)"
        summary.append(msg)
    print("[SUMMARY] " + " | ".join(summary))
    print(f"[INFO] Wrote: {rtr_path}, {label_path}, {sanity_path}, {paper_path}")


if __name__ == "__main__":
    main()
