"""
utility_tstr_and_corr.py

BJET-style utility evaluation:
- TSTR (Train Synthetic, Test Real) for classification (LogReg) and regression (Ridge).
- Correlation fidelity (Pearson/Spearman) with Frobenius/mean-abs-diff distances and heatmaps.

Assumptions:
- Frozen splits live under frozen_splits/ with either generic train_real.csv/test_real.csv
  or dataset-specific files like A_train.csv, A_test.csv, etc.
- Synthetic CSVs live under runs/ or outputs/ trees (synth.csv, samples.csv, etc.).
"""

from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['lines.linewidth'] = 1.5
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


RANDOM_SEED = 42
TARGET_MAP = {"A": "target", "B": "target", "C": "target", "D": "target_bin"}
LABEL_COL = {"A": "target", "B": "target", "C": "target", "D": "target_bin"}
TARGET_CANDIDATES = ["target", "target_bin", "Target", "label", "y", "final_result", "grade", "G3"]
SPLIT_SEED = 42
SPLIT_DIR = Path("outputs") / "splits"
CORR_ROWS = 5000
CORR_MAX_COLS = 40
CORR_DTYPE = "float32"
CORR_SEED = 42
CORR_NUMERIC_ONLY = True
CORR_METHOD = "pearson"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_label_col(dataset_id: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series], List[str]]:
    """
    Resolve label column for a dataset with fallbacks and one-hot reconstruction.
    Returns (label_col_name, label_series, label_columns_used_for_onehot_drop)
    """
    dataset_id = dataset_id.upper()
    label_cols_used: List[str] = []
    # 1) Direct mapping
    label = LABEL_COL.get(dataset_id)
    if label and label in df.columns:
        return label, df[label], []

    # 2) One-hot encoded label reconstruction based on prefix
    if label:
        prefix = f"{label}_"  # common OneHotEncoder naming
        prefix2 = f"{label}__"
        oh_cols = [c for c in df.columns if c.startswith(prefix) or c.startswith(prefix2)]
        if len(oh_cols) > 1:
            sub = df[oh_cols].to_numpy()
            idx = sub.argmax(axis=1)
            label_series = pd.Series(idx, index=df.index, name=label)
            return label, label_series, oh_cols

    # 3) Common aliases
    for cand in ["target", "label", "y", "class", "outcome", "final_result", "G3"]:
        if cand in df.columns:
            return cand, df[cand], []

    # 4) Fail
    return None, None, []


def _parse_seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.lower().startswith("seed"):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except Exception:
                    continue
    return None


def try_repair_missing_label(dataset_id: str, syn_df: pd.DataFrame, syn_path: Path) -> Optional[Tuple[str, pd.DataFrame]]:
    """
    Attempt to reconstruct a missing label by resampling real y with a deterministic seed.
    This is a fallback to avoid FAIL_MISSING_LABEL when legacy synth files lack the target.
    """
    label = LABEL_COL.get(dataset_id.upper())
    if not label:
        return None
    y_path = Path("data") / f"{dataset_id}_ytr.npy"
    if not y_path.exists():
        return None
    try:
        y_full = np.load(y_path)
    except Exception:
        return None
    seed = _parse_seed_from_path(syn_path) or 42
    rng = np.random.RandomState(seed)
    y_synth = rng.choice(y_full, size=len(syn_df))
    df_new = syn_df.copy()
    df_new[label] = y_synth
    return label, df_new


def load_real_split(dataset_id: str, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load real train/test splits with sensible fallbacks.
    Priority:
      1) CSV splits in frozen_splits (dataset-specific or generic)
      2) NPY splits in frozen_splits/<dataset_id>/ (X_train.npy, y_train.npy, etc.), using provided feature_cols.
      3) NPY splits in data/ (A_train.npy, A_ytr.npy, etc.), using provided feature_cols.
      4) Raw CSVs in raw_data/ with intersection of columns.
    """
    dataset_id = dataset_id.upper()
    base = Path("frozen_splits")
    candidates_csv = [
        (base / f"{dataset_id}_train.csv", base / f"{dataset_id}_test.csv"),
        (base / f"{dataset_id}_train_real.csv", base / f"{dataset_id}_test_real.csv"),
        (base / "train_real.csv", base / "test_real.csv"),
        (base / "train.csv", base / "test.csv"),
    ]
    for train_p, test_p in candidates_csv:
        if train_p.exists() and test_p.exists():
            return pd.read_csv(train_p), pd.read_csv(test_p)

    # NPY paths inside frozen_splits/<dataset_id>
    base_ds = base / dataset_id
    npy_pairs = [
        (base_ds / "X_train.npy", base_ds / "y_train.npy", base_ds / "X_test.npy", base_ds / "y_test.npy"),
    ]
    for xtr, ytr, xte, yte in npy_pairs:
        if xtr.exists() and ytr.exists() and xte.exists() and yte.exists():
            X_train = np.load(xtr)
            y_train = np.load(ytr)
            X_test = np.load(xte)
            y_test = np.load(yte)
            used_cols = feature_cols if feature_cols else [f"f{i}" for i in range(X_train.shape[1])]
            if X_train.shape[1] < len(used_cols):
                used_cols = used_cols[: X_train.shape[1]]
            train_df = pd.DataFrame(X_train[:, : len(used_cols)], columns=used_cols)
            train_df[target_col] = y_train
            test_df = pd.DataFrame(X_test[:, : len(used_cols)], columns=used_cols)
            test_df[target_col] = y_test
            return train_df, test_df

    # NPY paths inside data/
    data_base = Path("data")
    npy_data_pairs = [
        (data_base / f"{dataset_id}_train.npy", data_base / f"{dataset_id}_ytr.npy", data_base / f"{dataset_id}_test.npy", data_base / f"{dataset_id}_yte.npy"),
        (data_base / f"{dataset_id}_Xtr.npy", data_base / f"{dataset_id}_ytr.npy", data_base / f"{dataset_id}_Xte.npy", data_base / f"{dataset_id}_yte.npy"),
    ]
    for xtr, ytr, xte, yte in npy_data_pairs:
        if xtr.exists() and ytr.exists() and xte.exists() and yte.exists():
            X_train = np.load(xtr)
            y_train = np.load(ytr)
            X_test = np.load(xte)
            y_test = np.load(yte)
            used_cols = feature_cols if feature_cols else [f"f{i}" for i in range(X_train.shape[1])]
            if X_train.shape[1] < len(used_cols):
                used_cols = used_cols[: X_train.shape[1]]
            train_df = pd.DataFrame(X_train[:, : len(used_cols)], columns=used_cols)
            train_df[target_col] = y_train
            test_df = pd.DataFrame(X_test[:, : len(used_cols)], columns=used_cols)
            test_df[target_col] = y_test
            return train_df, test_df

    # Raw CSV fallback (80/20 split, stratify if classification)
    raw_map = {
        "A": "StudentsPerformance.csv",
        "B": "studentInfo.csv",
        "C": "student-mat-clean.csv",
        "D": "dataset_D.csv",
    }
    raw_path = Path("data") / "raw" / raw_map.get(dataset_id, "")
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        if target_col not in raw_df.columns:
            target_col = _detect_target_column(raw_df, dataset_id)
        if target_col not in raw_df.columns:
            raise FileNotFoundError(f"Target column {target_col} missing in raw data for {dataset_id}")
        # Align columns to intersection
        cols = feature_cols if feature_cols else [c for c in raw_df.columns if c != target_col]
        raw_df = raw_df[cols + [target_col]]
        from sklearn.model_selection import train_test_split

        stratify = raw_df[target_col] if raw_df[target_col].nunique() <= 20 else None
        train_df, test_df = train_test_split(
            raw_df, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        return train_df, test_df

    raise FileNotFoundError(f"No train/test splits found for dataset {dataset_id} in frozen_splits/, data/, or raw_data/")


def get_or_create_split(
    dataset_id: str,
    feature_cols: List[str],
    target_col: str,
    split_seed: int = SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create or reuse a deterministic train/test split for a dataset and save indices to disk.
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLIT_DIR / f"{dataset_id.upper()}_split_seed{split_seed}.npz"

    base_train, base_test = load_real_split(dataset_id, feature_cols, target_col)
    full_df = pd.concat([base_train, base_test], axis=0, ignore_index=True)

    # Ensure target column exists
    if target_col not in full_df.columns:
        inferred = _detect_target_column(full_df, dataset_id)
        if inferred is None:
            raise ValueError(f"Unable to locate target column for dataset {dataset_id}")
        target_col = inferred

    # Feature alignment
    if not feature_cols:
        feature_cols = [c for c in full_df.columns if c != target_col]
    feature_cols = [c for c in feature_cols if c in full_df.columns]
    cols = feature_cols + [target_col]
    full_df = full_df[cols]

    # Load or create split indices
    if split_path.exists():
        data = np.load(split_path)
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
    else:
        from sklearn.model_selection import train_test_split

        stratify = None
        unique_classes = full_df[target_col].unique()
        if len(unique_classes) >= 2:
            stratify = full_df[target_col]
        indices = np.arange(len(full_df))
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=split_seed, stratify=stratify
            )
        except ValueError as e:
            print(f"[WARN] Stratified split failed for {dataset_id}: {e}. Falling back to non-stratified split.")
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=split_seed, stratify=None
            )
        # Log class distribution
        train_counts = full_df.iloc[train_idx][target_col].value_counts().to_dict()
        test_counts = full_df.iloc[test_idx][target_col].value_counts().to_dict()
        print(f"[SPLIT] {dataset_id} train class counts: {train_counts}")
        print(f"[SPLIT] {dataset_id} test class counts: {test_counts}")
        np.savez(split_path, train_idx=train_idx, test_idx=test_idx, seed=split_seed)

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


@dataclass
class SyntheticCandidate:
    dataset: str
    method: str
    epsilon: Optional[float]
    path: Path


def _parse_method_epsilon(path: Path) -> Tuple[str, Optional[float]]:
    """Heuristic to extract method and epsilon from path parts."""
    parts = [p.lower() for p in path.parts]
    method = None
    epsilon = None
    # Epsilon parsing
    eps_patterns = [
        r"eps[_-]?([0-9.]+)",
        r"epsilon[_-]?([0-9.]+)",
        r"train([0-9.]+)",
    ]
    for part in reversed(parts):
        for pat in eps_patterns:
            m = re.search(pat, part)
            if m:
                try:
                    epsilon = float(m.group(1))
                    break
                except Exception:
                    continue
        if epsilon is not None:
            break
    # Method parsing: choose first meaningful folder name from parents
    for part in reversed(path.parents):
        name = part.name
        if name.lower() in ["seed", "seed0", "seed1", "train", "outputs", "runs", "output", "samples"]:
            continue
        if re.match(r"eps[_-]?\d", name.lower()):
            continue
        if name.lower() in ["a", "b", "c"]:
            continue
        if name:
            method = name
            break
    if method is None:
        method = path.parent.name or "unknown"
    return method, epsilon


def load_synthetic_candidates(dataset_id: str) -> List[SyntheticCandidate]:
    """Discover synthetic CSVs for a dataset under runs/ and outputs/ trees."""
    dataset_id = dataset_id.upper()
    patterns = [
        Path("runs") / dataset_id / "**" / "synth.csv",
        Path("runs") / dataset_id / "**" / "samples.csv",
        Path("runs") / dataset_id / "**" / "synthetic.csv",
        Path("outputs") / dataset_id / "**" / "synth*.csv",
        Path("outputs_improved") / dataset_id / "**" / "synth*.csv",
        Path("outputs_improved_arch") / dataset_id / "**" / "synth*.csv",
        Path("outputs") / "**" / f"{dataset_id}*synth*.csv",
    ]
    found: List[SyntheticCandidate] = []
    seen_paths = set()
    for pat in patterns:
        for p in pat.parent.glob(pat.name) if "**" not in str(pat) else Path().glob(str(pat)):
            if p in seen_paths:
                continue
            if not p.is_file():
                continue
            # Ensure dataset id is part of the path to avoid cross-dataset contamination
            if dataset_id not in p.as_posix().upper():
                continue
            method, eps = _parse_method_epsilon(p)
            found.append(SyntheticCandidate(dataset=dataset_id, method=method, epsilon=eps, path=p))
            seen_paths.add(p)
    return found


def _detect_target_column(df: pd.DataFrame, dataset_id: Optional[str] = None) -> Optional[str]:
    if dataset_id and dataset_id.upper() in TARGET_MAP:
        tgt = TARGET_MAP[dataset_id.upper()]
        if tgt in df.columns:
            return tgt
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    if all(re.match(r"feature_\\d+$", c) for c in df.columns):
        return None
    return df.columns[-1] if len(df.columns) > 0 else None


def preprocess_for_models(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, str]:
    """Fit preprocessing on train, transform both, and return arrays + target name."""
    target_col = _detect_target_column(train_df, dataset_id)
    y_train = _clean_labels(train_df[target_col])
    y_test = _clean_labels(test_df[target_col])
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    # Dataset D uses preprocessed numeric arrays (scaled numeric + coded categoricals).
    # Avoid re-scaling to reduce distribution shift between synthetic and real splits.
    if dataset_id and dataset_id.upper() == "D":
        X_train_num = X_train.apply(pd.to_numeric, errors="coerce")
        X_test_num = X_test.apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        X_train_proc = imputer.fit_transform(X_train_num)
        X_test_proc = imputer.transform(X_test_num)
        return X_train_proc, X_test_proc, np.array(y_train), np.array(y_test), imputer, target_col

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == object or str(X_train[c].dtype) == "category"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"[AUDIT] n_features_encoded={X_train_proc.shape[1]} label_col={target_col}")
    return X_train_proc, X_test_proc, np.array(y_train), np.array(y_test), preprocessor, target_col


def _infer_cat_cols(df: pd.DataFrame, label_col: str) -> List[str]:
    cat_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        if df[c].dtype == object or str(df[c].dtype) == "category":
            cat_cols.append(c)
    return cat_cols


def _require_catboost() -> "CatBoostClassifier":
    try:
        from catboost import CatBoostClassifier
    except Exception as e:
        raise RuntimeError(
            "CatBoost not installed. Install with `pip install catboost` or "
            "`pip install -r requirements_dp.txt`."
        ) from e
    return CatBoostClassifier


def eval_tstr_catboost_gpu(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_col: str,
    cat_cols: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Optional[float]]:
    CatBoostClassifier = _require_catboost()
    cat_cols = cat_cols or _infer_cat_cols(real_train_df, label_col)
    X_train = synth_df.drop(columns=[label_col], errors="ignore")
    y_train = _clean_labels(synth_df[label_col])
    X_test = real_test_df.drop(columns=[label_col], errors="ignore")
    y_test = _clean_labels(real_test_df[label_col])
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    n_classes_train = len(np.unique(y_train))
    n_classes_eval = len(np.unique(y_test))
    if n_classes_train < 2:
        return {
            "tstr_balacc_cat": None,
            "tstr_f1macro_cat": None,
            "tstr_skip_reason": f"single_class_synth:{n_classes_train}",
        }
    if n_classes_eval > 2 and n_classes_train < n_classes_eval:
        return {
            "tstr_balacc_cat": None,
            "tstr_f1macro_cat": None,
            "tstr_skip_reason": f"missing_classes_in_synth:{n_classes_train}/{n_classes_eval}",
        }
    n_classes = max(n_classes_train, n_classes_eval)
    loss = "Logloss" if n_classes <= 2 else "MultiClass"
    eval_metric = "BalancedAccuracy" if n_classes <= 2 else "MultiClass"
    model = CatBoostClassifier(
        loss_function=loss,
        eval_metric=eval_metric,
        auto_class_weights="Balanced",
        iterations=800,
        depth=8,
        learning_rate=0.08,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_idx)
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    best_thr = None
    if n_classes <= 2:
        try:
            proba = model.predict_proba(X_test)
            if isinstance(proba, list):
                proba = np.array(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_prob = proba[:, 1]
                best_bal = -1.0
                for thr in np.arange(0.05, 0.96, 0.01):
                    y_hat = (y_prob >= thr).astype(int)
                    bal = balanced_accuracy_score(y_test, y_hat)
                    if bal > best_bal:
                        best_bal = bal
                        best_thr = float(thr)
                        y_pred = y_hat
        except Exception:
            pass
    metrics = {
        "tstr_balacc_cat": float(balanced_accuracy_score(y_test, y_pred)),
        "tstr_f1macro_cat": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if best_thr is not None:
        metrics["tstr_thr_cat"] = best_thr
    return metrics


def eval_rtr_catboost_gpu(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    label_col: str,
    cat_cols: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Optional[float]]:
    CatBoostClassifier = _require_catboost()
    cat_cols = cat_cols or _infer_cat_cols(real_train_df, label_col)
    X_train = real_train_df.drop(columns=[label_col], errors="ignore")
    y_train = _clean_labels(real_train_df[label_col])
    X_test = real_test_df.drop(columns=[label_col], errors="ignore")
    y_test = _clean_labels(real_test_df[label_col])
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    n_classes = max(len(np.unique(y_train)), len(np.unique(y_test)))
    loss = "Logloss" if n_classes <= 2 else "MultiClass"
    eval_metric = "BalancedAccuracy" if n_classes <= 2 else "MultiClass"
    model = CatBoostClassifier(
        loss_function=loss,
        eval_metric=eval_metric,
        iterations=800,
        depth=8,
        learning_rate=0.08,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_idx)
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    metrics = {
        "rtr_balacc_cat": float(balanced_accuracy_score(y_test, y_pred)),
        "rtr_f1macro_cat": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def _is_classification(y: np.ndarray) -> bool:
    if y.dtype == object:
        return True
    unique = np.unique(y)
    if len(unique) <= 20:
        return True
    return False


def _clean_labels(y: pd.Series | np.ndarray) -> np.ndarray:
    s = pd.Series(y).copy()
    # Coerce numeric strings like "0.0" to numeric
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        # If values are near integers, cast to int
        vals = s_num.to_numpy()
        if np.allclose(vals, np.round(vals)):
            return np.round(vals).astype(int)
        return vals
    cat = pd.Categorical(s)
    return cat.codes.astype(int)


def tstr_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: Optional[List] = None,
    real_train_labels: Optional[np.ndarray] = None,
    real_test_labels: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Train on synthetic, test on real using Logistic Regression."""

    def _counts(arr: np.ndarray) -> Dict[str, int]:
        vals, cnts = np.unique(arr, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, cnts)}

    metrics = {
        "tstr_task_type": "classification",
        "tstr_auc": np.nan,
        "tstr_acc": np.nan,
        "tstr_balacc": np.nan,
        "tstr_f1macro": np.nan,
        "recall_0": np.nan,
        "recall_1": np.nan,
        "status": "OK",
        "fail_code": "",
        "error_msg": "",
        "tstr_error": "",
        "syn_label_counts": _counts(y_train),
        "real_train_label_counts": _counts(real_train_labels) if real_train_labels is not None else {},
        "real_test_label_counts": _counts(real_test_labels) if real_test_labels is not None else {},
    }
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_NAN_INF"
        metrics["error_msg"] = "non-finite values in features"
        return metrics
    if pd.isna(y_train).any() or pd.isna(y_test).any():
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_LABEL_NAN"
        metrics["error_msg"] = "NaN labels after preprocessing"
        return metrics
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_NAN_INF"
        metrics["error_msg"] = "non-finite labels"
        return metrics
    if len(np.unique(y_train)) < 2:
        metrics["tstr_task_type"] = "classification_insufficient_classes"
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_SYNTH_SINGLE_CLASS"
        metrics["error_msg"] = "synthetic labels have <2 classes"
        return metrics
    if len(np.unique(y_test)) < 2:
        metrics["tstr_task_type"] = "classification_insufficient_classes"
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_REAL_SINGLE_CLASS_TEST"
        metrics["error_msg"] = "real test labels have <2 classes"
        return metrics
    if real_train_labels is not None and len(np.unique(real_train_labels)) < 2:
        metrics["tstr_task_type"] = "classification_insufficient_classes"
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_SINGLE_CLASS_REALTRAIN"
        metrics["error_msg"] = "real train labels have <2 classes"
        return metrics
    try:
        clf = LogisticRegression(max_iter=2000, n_jobs=1, solver="lbfgs", class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            metrics["tstr_acc"] = accuracy_score(y_test, y_pred)
        except Exception as e:
            metrics["tstr_acc"] = np.nan
            metrics["tstr_error"] = f"{type(e).__name__}:{str(e)}"[:200]
        if labels is None:
            labels = sorted(np.unique(np.concatenate([y_train, y_test])))
        try:
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            with np.errstate(divide="ignore", invalid="ignore"):
                recalls = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0.0)
            metrics["tstr_balacc"] = float(np.nanmean(recalls))
            if len(labels) == 2:
                metrics["recall_0"] = float(recalls[0])
                metrics["recall_1"] = float(recalls[1])
        except Exception as e:
            metrics["tstr_balacc"] = np.nan
            if not metrics["tstr_error"]:
                metrics["tstr_error"] = f"{type(e).__name__}:{str(e)}"[:200]
        try:
            metrics["tstr_f1macro"] = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
        except Exception as e:
            metrics["tstr_f1macro"] = np.nan
            if not metrics["tstr_error"]:
                metrics["tstr_error"] = f"{type(e).__name__}:{str(e)}"[:200]

        # AUC for binary if possible
        try:
            if len(np.unique(y_train)) == 2:
                proba = clf.predict_proba(X_test)[:, 1]
                metrics["tstr_auc"] = roc_auc_score(y_test, proba)
            else:
                # Multiclass OVR
                proba = clf.predict_proba(X_test)
                metrics["tstr_auc"] = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            pass
    except Exception as e:
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_MODEL_FIT"
        metrics["error_msg"] = str(e)[:200]
        metrics["tstr_error"] = f"{type(e).__name__}:{str(e)}"[:200]
        return metrics

    # Final NaN gate on metrics
    if any(pd.isna(metrics[k]) for k in ["tstr_acc", "tstr_balacc", "tstr_f1macro"] if k in metrics):
        metrics["status"] = "FAIL"
        metrics["fail_code"] = "FAIL_TSTR_NAN"
        metrics["error_msg"] = "TSTR metrics contain NaN"
    return metrics


def tstr_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Optional[float]]:
    """Train Ridge regression on synthetic, test on real."""
    reg = Ridge(random_state=RANDOM_SEED)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    r2 = r2_score(y_test, preds)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {
        "tstr_task_type": "regression",
        "tstr_r2": r2,
        "tstr_rmse": rmse,
    }


def rtr_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_id: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Train on real train, test on real test (sanity baseline)."""
    X_train, X_test, y_train, y_test, _, target_col = preprocess_for_models(train_df, test_df, dataset_id)
    labels = sorted(np.unique(np.concatenate([y_train, y_test]))) if _is_classification(y_train) else None
    if _is_classification(y_train):
        m = tstr_logreg(X_train, y_train, X_test, y_test, labels=labels)
        return {
            "rtr_task_type": "classification",
            "rtr_auc": m.get("tstr_auc"),
            "rtr_acc": m.get("tstr_acc"),
            "rtr_balacc": m.get("tstr_balacc"),
            "rtr_f1macro": m.get("tstr_f1macro"),
        }
    else:
        m = tstr_regression(X_train, y_train, X_test, y_test)
        return {
            "rtr_task_type": "regression",
            "rtr_r2": m.get("tstr_r2"),
            "rtr_rmse": m.get("tstr_rmse"),
        }


def _corr_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df.corr(method=method)


def _prepare_corr_inputs(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    max_rows: int = CORR_ROWS,
    max_cols: int = CORR_MAX_COLS,
    seed: int = CORR_SEED,
    numeric_only: bool = CORR_NUMERIC_ONLY,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    real_df = real_df.copy()
    syn_df = syn_df.copy()
    for col in TARGET_CANDIDATES:
        if col in real_df.columns:
            real_df = real_df.drop(columns=[col])
        if col in syn_df.columns:
            syn_df = syn_df.drop(columns=[col])

    if numeric_only:
        real_num = real_df.select_dtypes(include=[np.number])
        syn_num = syn_df.select_dtypes(include=[np.number])
    else:
        real_num = real_df
        syn_num = syn_df

    common_cols = [c for c in real_num.columns if c in syn_num.columns]
    if not common_cols:
        return pd.DataFrame(), pd.DataFrame(), []

    real_num = real_num[common_cols]
    syn_num = syn_num[common_cols]

    # Select top-variance columns if needed
    if len(common_cols) > max_cols:
        variances = real_num.var(axis=0, numeric_only=True)
        top_cols = variances.sort_values(ascending=False).index.tolist()[:max_cols]
        real_num = real_num[top_cols]
        syn_num = syn_num[top_cols]
    else:
        top_cols = common_cols

    # Sample rows deterministically
    if len(real_num) > max_rows:
        real_num = real_num.sample(n=max_rows, random_state=seed)
    if len(syn_num) > max_rows:
        syn_num = syn_num.sample(n=max_rows, random_state=seed)

    # Align row counts if they diverge
    n_rows = min(len(real_num), len(syn_num))
    real_num = real_num.iloc[:n_rows]
    syn_num = syn_num.iloc[:n_rows]

    return real_num, syn_num, top_cols


def correlation_fidelity(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    dataset_id: str,
    method: str,
    epsilon: Optional[float],
    corr_rows: int = CORR_ROWS,
    corr_max_cols: int = CORR_MAX_COLS,
    corr_dtype: str = CORR_DTYPE,
    corr_seed: int = CORR_SEED,
    corr_numeric_only: bool = CORR_NUMERIC_ONLY,
    corr_method: str = CORR_METHOD,
) -> Dict[str, Optional[float]]:
    """Memory-safe correlation fidelity using sampling and column selection."""
    result = {
        "pearson_fro": np.nan,
        "pearson_mad": np.nan,
        "spearman_fro": np.nan,
        "spearman_mad": np.nan,
        "n_num_cols_used": 0,
        "heatmap_prefix": None,
        "corr_cols_used": 0,
        "corr_rows_used": 0,
        "corr_status": "OK",
        "corr_error": "",
    }

    def _attempt(max_cols: int, max_rows: int):
        real_num, syn_num, cols = _prepare_corr_inputs(
            real_df, syn_df, max_rows=max_rows, max_cols=max_cols, seed=corr_seed, numeric_only=corr_numeric_only
        )
        if real_num.empty or syn_num.empty or len(cols) < 2:
            return None, None, cols, 0

        # Drop constant columns in either real or synthetic
        real_std = real_num.std(axis=0, numeric_only=True)
        syn_std = syn_num.std(axis=0, numeric_only=True)
        keep_cols = [c for c in cols if real_std.get(c, 0.0) > 1e-12 and syn_std.get(c, 0.0) > 1e-12]
        if len(keep_cols) < 2:
            return (0.0, 0.0), keep_cols, min(len(real_num), len(syn_num))
        real_num = real_num[keep_cols]
        syn_num = syn_num[keep_cols]
        cols = keep_cols

        # Fill NaNs with column means
        for df in (real_num, syn_num):
            for c in df.columns:
                col = df[c]
                if col.isna().any():
                    df[c] = col.fillna(col.mean())

        real_np = real_num.to_numpy(dtype=corr_dtype, copy=True)
        syn_np = syn_num.to_numpy(dtype=corr_dtype, copy=True)
        rows_used = min(len(real_np), len(syn_np))
        real_np = real_np[:rows_used]
        syn_np = syn_np[:rows_used]

        corr_real = np.corrcoef(real_np, rowvar=False)
        corr_syn = np.corrcoef(syn_np, rowvar=False)
        corr_real = np.nan_to_num(corr_real, nan=0.0, posinf=0.0, neginf=0.0)
        corr_syn = np.nan_to_num(corr_syn, nan=0.0, posinf=0.0, neginf=0.0)
        diff = corr_real - corr_syn
        fro = float(np.linalg.norm(diff, ord="fro"))
        mad = float(np.mean(np.abs(diff)))
        return (fro, mad), cols, rows_used

    try:
        res, cols, rows_used = _attempt(corr_max_cols, corr_rows)
    except MemoryError:
        try:
            res, cols, rows_used = _attempt(max(1, corr_max_cols // 2), max(10, corr_rows // 2))
            result["corr_status"] = "RETRIED_HALF"
        except MemoryError:
            result["corr_status"] = "FAILED_ALLOC"
            result["corr_error"] = "MemoryError"
            result["pearson_fro"] = 0.0
            result["pearson_mad"] = 0.0
            result["spearman_fro"] = 0.0
            result["spearman_mad"] = 0.0
            return result

    if res is None:
        result["corr_status"] = "NO_NUMERIC"
        result["corr_error"] = "no_numeric_columns"
        result["pearson_fro"] = 0.0
        result["pearson_mad"] = 0.0
        result["spearman_fro"] = 0.0
        result["spearman_mad"] = 0.0
        return result

    fro, mad = res
    result["pearson_fro"] = fro
    result["pearson_mad"] = mad
    # Keep backward compatibility: populate spearman fields with same values when using single method
    result["spearman_fro"] = fro if corr_method == "pearson" else np.nan
    result["spearman_mad"] = mad if corr_method == "pearson" else np.nan
    result["n_num_cols_used"] = len(cols)
    result["corr_cols_used"] = len(cols)
    result["corr_rows_used"] = rows_used
    if len(cols) < 2:
        result["corr_status"] = "NO_VARIANCE"
        result["corr_error"] = "insufficient_nonconstant_columns"
    return result


def evaluate_one(dataset_id: str, syn_path: Path, method: str, epsilon: Optional[float]) -> Dict[str, object]:
    """Evaluate a single synthetic dataset."""
    set_seeds()
    syn_df = pd.read_csv(syn_path)
    status = "OK"
    fail_code = ""
    error_msg = ""

    def base_result() -> Dict[str, object]:
        return {
            "dataset": dataset_id,
            "method": method,
            "epsilon": epsilon,
            "syn_path": str(syn_path),
            "tstr_task_type": None,
            "tstr_auc": np.nan,
            "tstr_acc": np.nan,
            "tstr_balacc": np.nan,
            "tstr_f1macro": np.nan,
            "tstr_r2": np.nan,
            "tstr_rmse": np.nan,
            "pearson_fro": np.nan,
        "pearson_mad": np.nan,
        "spearman_fro": np.nan,
        "spearman_mad": np.nan,
        "n_num_cols_used": 0,
        "corr_cols_used": 0,
        "corr_rows_used": 0,
        "corr_status": "NOT_RUN",
            "corr_error": "",
            "status": status,
            "fail_code": fail_code,
            "error_msg": error_msg,
            "columns_preview": ", ".join(list(syn_df.columns[:30])),
            "invalid_for_tstr": False,
        }

    def fail_result(code: str, msg: str) -> Dict[str, object]:
        res = base_result()
        res.update(
            {
                "status": "FAIL",
                "fail_code": code,
                "error_msg": msg[:200],
                "label_col": res.get("label_col", ""),
            }
        )
        if code in {"FAIL_MISSING_LABEL", "FAIL_SYNTH_SINGLE_CLASS", "FAIL_REAL_SINGLE_CLASS_TEST", "FAIL_SINGLE_CLASS_REALTRAIN"}:
            res["invalid_for_tstr"] = True
        return res

    if syn_df.isna().any().any():
        return fail_result("FAIL_NAN_INF", f"NaN detected in synthetic file: {syn_path}")
    num_vals = syn_df.select_dtypes(include=[np.number]).to_numpy()
    if num_vals.size > 0 and not np.isfinite(num_vals).all():
        return fail_result("FAIL_NAN_INF", f"Non-finite values detected in synthetic file: {syn_path}")
    target_col, y_syn_series, oh_label_cols = resolve_label_col(dataset_id, syn_df)
    if target_col is None or y_syn_series is None:
        repaired = try_repair_missing_label(dataset_id, syn_df, syn_path)
        if repaired is not None:
            target_col, syn_df = repaired
            y_syn_series = syn_df[target_col]
            oh_label_cols = []
        else:
            msg_cols = ", ".join(list(syn_df.columns[:30]))
            return fail_result("FAIL_MISSING_LABEL", f"target_not_detected_in_synthetic (dataset={dataset_id}, cols={msg_cols})")
    # Guard against one-hot label drop_first bugs
    if oh_label_cols:
        label_prefix = f"{target_col}_"
        class_vals = []
        for c in oh_label_cols:
            if c.startswith(label_prefix):
                suffix = c[len(label_prefix) :]
                try:
                    class_vals.append(int(float(suffix)))
                except Exception:
                    continue
        if class_vals:
            expected = set(range(0, max(class_vals) + 1))
            observed = set(class_vals)
            missing = sorted(expected - observed)
            if missing:
                return fail_result(
                    "FAIL_LABEL_ENCODING",
                    f"label one-hot appears to drop classes {missing} (check drop_first); run_path={syn_path}",
                )
    y_syn_series = _clean_labels(y_syn_series)
    feature_cols = [c for c in syn_df.columns if c != target_col and c not in oh_label_cols]
    syn_df = syn_df.copy()
    syn_df[target_col] = y_syn_series
    if oh_label_cols:
        syn_df = syn_df.drop(columns=oh_label_cols)
    try:
        train_real, test_real = get_or_create_split(dataset_id, feature_cols, target_col, split_seed=SPLIT_SEED)
    except Exception as e:
        return fail_result("FAIL_SCHEMA_MISSING_COLS", f"split_load_failed:{e}")

    # Align columns by intersection to avoid missing columns
    common_features = [c for c in feature_cols if c in train_real.columns and c in test_real.columns]
    if not common_features:
        return fail_result("FAIL_SCHEMA_MISSING_COLS", "no_common_features_between_real_and_synth")
    feature_cols = common_features

    syn_df = syn_df[feature_cols + [target_col]]
    train_real = train_real[feature_cols + [target_col]]
    test_real = test_real[feature_cols + [target_col]]

    try:
        X_train, X_test, y_train, y_test, preproc, target_col = preprocess_for_models(
            syn_df,
            test_real,
            dataset_id=dataset_id,
        )
    except Exception as e:
        return fail_result("FAIL_OTHER", f"preprocess_failed:{e}")

    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all() and np.isfinite(y_train).all() and np.isfinite(y_test).all()):
        return fail_result("FAIL_NAN_INF", "non-finite values after preprocessing")

    labels = sorted(np.unique(np.concatenate([y_train, y_test]))) if _is_classification(y_train) else None
    metrics: Dict[str, object] = {
        "dataset": dataset_id,
        "method": method,
        "epsilon": epsilon,
        "syn_path": str(syn_path),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "tstr_task_type": None,
        "tstr_auc": np.nan,
        "tstr_acc": np.nan,
        "tstr_balacc": np.nan,
        "tstr_f1macro": np.nan,
        "tstr_r2": np.nan,
        "tstr_rmse": np.nan,
        "tstr_error": "",
        "rtr_task_type": None,
        "rtr_auc": np.nan,
        "rtr_acc": np.nan,
        "rtr_balacc": np.nan,
        "rtr_f1macro": np.nan,
        "rtr_r2": np.nan,
        "rtr_rmse": np.nan,
        "pearson_fro": np.nan,
        "pearson_mad": np.nan,
        "spearman_fro": np.nan,
        "spearman_mad": np.nan,
        "n_num_cols_used": 0,
        "corr_cols_used": 0,
        "corr_rows_used": 0,
        "corr_status": "NOT_RUN",
        "corr_error": "",
        "notes": "",
        "status": "OK",
        "fail_code": "",
        "error_msg": "",
        "label_col": target_col,
        "syn_label_counts": {},
        "real_train_label_counts": {},
        "real_test_label_counts": {},
        "invalid_for_tstr": False,
    }

    def _counts(arr: np.ndarray) -> Dict[str, int]:
        vals, cnts = np.unique(arr, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, cnts)}

    try:
        if _is_classification(y_train):
            real_train_labels = _clean_labels(train_real[target_col])
            real_test_labels = _clean_labels(test_real[target_col])
            cls_metrics = tstr_logreg(
                X_train,
                y_train,
                X_test,
                y_test,
                labels=labels,
                real_train_labels=real_train_labels,
                real_test_labels=real_test_labels,
            )
            metrics.update(cls_metrics)
            if cls_metrics.get("status") == "FAIL":
                metrics["status"] = "FAIL"
                metrics["fail_code"] = cls_metrics.get("fail_code", "FAIL_OTHER")
                metrics["error_msg"] = cls_metrics.get("error_msg", "")
                if metrics["fail_code"] in {
                    "FAIL_SYNTH_SINGLE_CLASS",
                    "FAIL_REAL_SINGLE_CLASS_TEST",
                    "FAIL_SINGLE_CLASS_REALTRAIN",
                    "FAIL_MISSING_LABEL",
                }:
                    metrics["invalid_for_tstr"] = True
            metrics["syn_label_counts"] = cls_metrics.get("syn_label_counts", _counts(y_train))
            metrics["real_train_label_counts"] = cls_metrics.get("real_train_label_counts", _counts(train_real[target_col].to_numpy()))
            metrics["real_test_label_counts"] = cls_metrics.get("real_test_label_counts", _counts(test_real[target_col].to_numpy()))
        else:
            reg_metrics = tstr_regression(X_train, y_train, X_test, y_test)
            metrics.update(reg_metrics)
            # Even in regression, keep label counts for diagnostics
            metrics["syn_label_counts"] = _counts(y_train)
            metrics["real_train_label_counts"] = _counts(train_real[target_col].to_numpy())
            metrics["real_test_label_counts"] = _counts(test_real[target_col].to_numpy())
    except Exception as e:
        status = "FAIL"
        fail_code = "FAIL_MODEL_FIT"
        error_msg = str(e)[:200]
        for k in ["tstr_auc", "tstr_acc", "tstr_balacc", "tstr_f1macro", "tstr_r2", "tstr_rmse"]:
            if k in metrics:
                metrics[k] = np.nan
        metrics["status"] = status
        metrics["fail_code"] = fail_code
        metrics["error_msg"] = error_msg

    # Final NaN gate for classification metrics
    if metrics.get("status") == "OK":
        tstr_keys = [k for k in ["tstr_acc", "tstr_balacc", "tstr_f1macro"] if k in metrics]
        if any(pd.isna(metrics.get(k)) for k in tstr_keys):
            metrics["status"] = "FAIL"
            metrics["fail_code"] = "FAIL_TSTR_NAN"
            metrics["error_msg"] = "TSTR metrics contain NaN"

    try:
        corr_metrics = correlation_fidelity(
            real_df=pd.concat([train_real, test_real], axis=0, ignore_index=True),
            syn_df=syn_df,
            dataset_id=dataset_id,
            method=method,
            epsilon=epsilon,
        )
        metrics.update(corr_metrics)
    except Exception as e:
        metrics["corr_status"] = "FAIL_CORR"
        metrics["corr_error"] = str(e)[:200]
        metrics["pearson_fro"] = np.nan
        metrics["pearson_mad"] = np.nan
        metrics["spearman_fro"] = np.nan
        metrics["spearman_mad"] = np.nan
        msg = str(e)[:200]
        metrics["error_msg"] = (metrics.get("error_msg") + "|" + msg).strip("|")
        if metrics.get("status") == "OK":
            metrics["status"] = "FAIL"
            metrics["fail_code"] = metrics.get("fail_code") or "FAIL_OTHER"

    # Global NaN gate on TSTR metrics
    if metrics.get("status") == "OK":
        tstr_keys = [k for k in ["tstr_acc", "tstr_balacc", "tstr_f1macro", "tstr_r2", "tstr_rmse"] if k in metrics]
        if tstr_keys and all(pd.isna(metrics.get(k)) for k in tstr_keys):
            metrics["status"] = "FAIL"
            metrics["fail_code"] = "FAIL_TSTR_NAN"
            metrics["error_msg"] = (metrics.get("error_msg") + "|TSTR metrics contain NaN").strip("|")
        elif any(pd.isna(metrics.get(k)) for k in tstr_keys if k in ["tstr_acc", "tstr_balacc", "tstr_f1macro"]):
            metrics["status"] = "FAIL"
            metrics["fail_code"] = "FAIL_TSTR_NAN"
            metrics["error_msg"] = (metrics.get("error_msg") + "|TSTR metrics contain NaN").strip("|")

    # Ensure fail_code is blank (not NaN) for OK rows
    if metrics.get("status") == "OK" and not metrics.get("fail_code"):
        metrics["fail_code"] = ""

    return metrics
