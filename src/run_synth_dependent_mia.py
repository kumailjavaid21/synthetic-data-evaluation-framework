import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
tstr_path = ROOT / "run_tstr_and_corr_eval.py"
spec = importlib.util.spec_from_file_location("tstr_eval", str(tstr_path))
tstr_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tstr_eval)

load_real_split = tstr_eval.util.load_real_split
resolve_label_col = tstr_eval.resolve_label_col


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_feature_names(dataset: str) -> Optional[List[str]]:
    meta_path = Path("data") / "processed" / dataset.upper() / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = meta.get("feature_names") or meta.get("ohe_feature_names")
        if isinstance(names, list) and names:
            return names
    except Exception:
        return None
    return None


def _load_real_arrays(dataset: str, feature_names: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path("data")
    xtr = data_dir / f"{dataset}_Xtr.npy"
    ytr = data_dir / f"{dataset}_ytr.npy"
    xte = data_dir / f"{dataset}_Xte.npy"
    yte = data_dir / f"{dataset}_yte.npy"
    if xtr.exists() and ytr.exists() and xte.exists() and yte.exists():
        return np.load(xtr), np.load(xte)

    # Fallback: use load_real_split and numeric conversion
    dummy_cols = feature_names or []
    train_df, test_df = load_real_split(dataset, dummy_cols, target_col="target")
    label_col, _, _ = resolve_label_col(dataset, train_df)
    if label_col is None:
        raise ValueError(f"Unable to detect label column for dataset {dataset}")
    X_train = train_df.drop(columns=[label_col], errors="ignore")
    X_test = test_df.drop(columns=[label_col], errors="ignore")
    X_train = pd.get_dummies(X_train, drop_first=False)
    X_test = pd.get_dummies(X_test, drop_first=False)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)
    return X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)


def _align_synth(synth_df: pd.DataFrame, feature_names: Optional[List[str]], target_col: Optional[str]) -> np.ndarray:
    df = synth_df.copy()
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    if feature_names and all(c in df.columns for c in feature_names):
        df = df[feature_names]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.to_numpy(dtype=np.float32, copy=False)


def _min_l2_distances(x: np.ndarray, s: np.ndarray, chunk: int = 1024) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    s = s.astype(np.float32, copy=False)
    s2 = np.sum(s * s, axis=1)
    out = np.empty(x.shape[0], dtype=np.float32)
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk]
        xb2 = np.sum(xb * xb, axis=1)[:, None]
        d2 = xb2 + s2[None, :] - 2.0 * (xb @ s.T)
        d2 = np.maximum(d2, 0.0)
        out[i : i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs_A_IMPROVE")
    parser.add_argument("--datasets", nargs="+", default=["A"])
    parser.add_argument("--out_csv", type=str, default="outputs_A_IMPROVE/mia_synthdep_results.csv")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    rows: List[Dict[str, object]] = []
    auc_vals = []
    hashes = []

    for ds in args.datasets:
        ds_upper = ds.upper()
        feature_names = None
        try:
            feature_names = _load_feature_names(ds_upper)
        except Exception:
            feature_names = None
        X_members, X_nonmembers = _load_real_arrays(ds_upper, feature_names)

        synth_paths = list((outputs_root / ds_upper).rglob("synth.csv"))
        synth_paths += list((outputs_root / ds_upper).rglob("synth_unrepaired.csv"))
        if not synth_paths:
            print(f"[WARN] No synth files found for dataset {ds_upper} in {outputs_root}")
            continue

        rng = np.random.RandomState(42)
        for sp in synth_paths:
            try:
                synth_df = pd.read_csv(sp)
            except Exception as e:
                print(f"[WARN] Failed to read {sp}: {e}")
                continue
            label_col, _, _ = resolve_label_col(ds_upper, synth_df)
            synth_arr = _align_synth(synth_df, feature_names, label_col)
            if synth_arr.shape[1] != X_members.shape[1]:
                print(f"[WARN] Feature mismatch for {sp}: synth={synth_arr.shape[1]} real={X_members.shape[1]}")
                continue

            if synth_arr.shape[0] > 5000:
                synth_arr = synth_arr[rng.choice(synth_arr.shape[0], size=5000, replace=False)]
            synth_sample = synth_arr
            if synth_sample.shape[0] > 2000:
                synth_sample = synth_sample[rng.choice(synth_sample.shape[0], size=2000, replace=False)]

            scores_members = -_min_l2_distances(X_members, synth_sample)
            scores_nonmembers = -_min_l2_distances(X_nonmembers, synth_sample)
            scores = np.concatenate([scores_members, scores_nonmembers])
            labels = np.concatenate([np.ones(len(scores_members)), np.zeros(len(scores_nonmembers))])
            auc = roc_auc_score(labels, scores)

            syn_hash = _md5(sp)
            print(f"[MIA-SYNTHDEP] run={sp.parent} syn={sp.name} auc={auc:.4f}")
            rows.append(
                {
                    "dataset": ds_upper,
                    "run_dir": str(sp.parent),
                    "syn_path": str(sp),
                    "mia_auc_synthdep": float(auc),
                    "n_members": int(len(X_members)),
                    "n_nonmembers": int(len(X_nonmembers)),
                    "n_synth": int(len(synth_arr)),
                    "syn_hash": syn_hash,
                }
            )
            auc_vals.append(float(auc))
            hashes.append(syn_hash)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")

    if rows:
        vals = np.array(auc_vals, dtype=float)
        uniq = int(np.unique(np.round(vals, 12)).size)
        print(
            f"[STAT] rows={len(rows)} min={np.min(vals):.6f} "
            f"median={np.median(vals):.6f} max={np.max(vals):.6f} unique={uniq}"
        )
    if auc_vals:
        vals = np.array(auc_vals, dtype=float)
        print(
            f"[STAT] mia_auc_synthdep min={np.min(vals):.6f} "
            f"median={np.median(vals):.6f} max={np.max(vals):.6f}"
        )
        uniq_hashes = len(set(hashes))
        if uniq_hashes > 1 and (np.max(vals) - np.min(vals) <= 1e-9):
            print("WARNING: MIA AUC constant across different synth hashes.")


if __name__ == "__main__":
    main()
