import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utility_tstr_and_corr import (
    resolve_label_col,
    get_or_create_split,
    SPLIT_SEED,
    try_repair_missing_label,
)


RNG = np.random.default_rng(0)
MAX_SYNTH = 5000
CHUNK = 256
DEFAULT_K = 5
LEAKY_NAME_TOKENS = ["student", "user", "index", "uid", "guid", "email"]
LABEL_TOKENS = ["target", "label", "y", "target_bin", "outcome"]


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


def _infer_dataset(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.upper() in {"A", "B", "C", "D"}:
            return part.upper()
    return None


def _parse_seed(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.lower().startswith("seed"):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _parse_method(path: Path) -> str:
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


def _parse_eps(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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


def _load_synth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def drop_leaky_cols(df: pd.DataFrame, ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty:
        return df
    cols = list(df.columns)
    drop = []
    for c in cols:
        c_l = str(c).lower()
        if any(tok in c_l for tok in LEAKY_NAME_TOKENS):
            drop.append(c)
        if any(tok == c_l for tok in LABEL_TOKENS):
            drop.append(c)
        if re.search(r"(^|_)(id|student_id|user_id)($|_)", c_l):
            drop.append(c)
    ref = ref_df if ref_df is not None else df
    n_rows = len(ref)
    if n_rows > 0:
        for c in cols:
            if c in drop:
                continue
            if c not in ref.columns:
                continue
            nunique = ref[c].nunique(dropna=False)
            uniq_vals = pd.Series(ref[c]).dropna().unique()
            if len(uniq_vals) > 0 and set(uniq_vals).issubset({0, 1, 0.0, 1.0}):
                continue
            if nunique <= 2:
                continue
            if nunique > 100 and (nunique / n_rows) > 0.99:
                drop.append(c)
    drop = sorted(set(drop))
    if drop:
        print(f"[KNN] Dropped leaky cols: {drop}")
    return df.drop(columns=drop, errors="ignore")


def _prepare_features(
    real_train: pd.DataFrame, real_test: pd.DataFrame, synth: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _encode(rt: pd.DataFrame, rtest: pd.DataFrame, syn: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        common = set(rt.columns) & set(rtest.columns) & set(syn.columns)
        if not common:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
        common = sorted(common)
        rt = rt[common]
        rtest = rtest[common]
        syn = syn[common]

        combined = pd.concat([rt, rtest, syn], axis=0, ignore_index=True)
        combined = pd.get_dummies(combined, drop_first=False)

        n_train = len(rt)
        n_test = len(rtest)
        enc_train = combined.iloc[:n_train].to_numpy(dtype=np.float32)
        enc_test = combined.iloc[n_train : n_train + n_test].to_numpy(dtype=np.float32)
        enc_synth = combined.iloc[n_train + n_test :].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        scaler.fit(enc_train)
        enc_train = scaler.transform(enc_train)
        enc_test = scaler.transform(enc_test)
        enc_synth = scaler.transform(enc_synth)
        return enc_train, enc_test, enc_synth

    rt_drop = drop_leaky_cols(real_train)
    rtest_drop = drop_leaky_cols(real_test, ref_df=rt_drop)
    syn_drop = drop_leaky_cols(synth, ref_df=rt_drop)
    enc_train, enc_test, enc_synth = _encode(rt_drop, rtest_drop, syn_drop)
    if enc_train.shape[1] >= 5:
        return enc_train, enc_test, enc_synth
    # Fallback: keep more features if leaky-drop leaves too few.
    return _encode(real_train, real_test, synth)


def _load_processed_split(dataset: str, label_col: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    processed_dir = Path("data") / "processed" / dataset.upper()
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        if label_col in train_df.columns and label_col in test_df.columns:
            return train_df, test_df
    return None


def _knn_scores(x: np.ndarray, synth: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or synth.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(k, synth.shape[0]), metric="euclidean", algorithm="auto")
    nn.fit(synth)
    dists, _ = nn.kneighbors(x, return_distance=True)
    mean_dist = dists.mean(axis=1)
    scores = -mean_dist.astype(np.float32)
    return scores, mean_dist.astype(np.float32)


def compute_knn_mia_for_one(
    path: Path,
    k: int,
    generated_at: str,
    sanity: int,
    debug: bool,
    want_sanity_payload: bool,
) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
    try:
        df_synth = _load_synth(path)
        dataset = _infer_dataset(path)
        if dataset is None:
            raise ValueError("dataset_not_found_in_path")
        label_col, _, _ = resolve_label_col(dataset, df_synth)
        if label_col is None:
            repaired = try_repair_missing_label(dataset, df_synth, path)
            if repaired is None:
                raise ValueError("label_col_not_found")
            label_col, df_synth = repaired
        synth_features = df_synth.drop(columns=[label_col], errors="ignore")

        processed_split = _load_processed_split(dataset, label_col)
        if processed_split is not None:
            train_real, test_real = processed_split
        else:
            if dataset.upper() == "B":
                raise ValueError("processed_split_missing")
            train_real, test_real = get_or_create_split(
                dataset, list(synth_features.columns), label_col, split_seed=SPLIT_SEED
            )
        real_train = train_real.drop(columns=[label_col], errors="ignore")
        real_test = test_real.drop(columns=[label_col], errors="ignore")

        if len(real_test) == 0:
            raise ValueError("no_real_test_rows")

        n_members = len(real_train)
        n_nonmembers = min(len(real_test), n_members)
        non_idx = RNG.choice(len(real_test), size=n_nonmembers, replace=False)
        real_test = real_test.iloc[non_idx]

        enc_train, enc_test, enc_synth = _prepare_features(real_train, real_test, synth_features)
        if enc_train.shape[1] < 5:
            return {
                "run_dir": _normalize_dir_key(str(path.parent)),
                "syn_path": _normalize_path(str(path)),
                "dataset": dataset,
                "method": _parse_method(path),
                "seed": _parse_seed(path),
                "epsilon": _parse_eps(path)[0],
                "epsilon_train": _parse_eps(path)[1],
                "epsilon_repair": _parse_eps(path)[2],
                "knn_mia_auc_raw": np.nan,
                "knn_mia_auc": np.nan,
                "auc_flipped": np.nan,
                "k": k,
                "n_members": np.nan,
                "n_nonmembers": np.nan,
                "repaired_flag": np.nan,
                "synth_kind": "synth_unrepaired" if path.name.lower() == "synth_unrepaired.csv" else "synth",
                "status": "SKIP",
                "fail_code": "low_dim",
                "error_msg": "insufficient_feature_dim",
                "generated_at": generated_at,
                "run_dir_key": _normalize_dir_key(str(path.parent)),
                "syn_path_key": _normalize_path(str(path)),
            }, None, None
        if enc_synth.shape[0] > MAX_SYNTH:
            idx = RNG.choice(enc_synth.shape[0], size=MAX_SYNTH, replace=False)
            enc_synth = enc_synth[idx]

        assert enc_train is not enc_test
        assert enc_train.shape[0] > 10 and enc_test.shape[0] > 10
        assert enc_synth.shape[0] > 10

        k = min(k, enc_synth.shape[0])
        scores_members, d_members = _knn_scores(enc_train, enc_synth, k=k)
        scores_non, d_non = _knn_scores(enc_test, enc_synth, k=k)
        labels = np.concatenate([np.ones_like(scores_members), np.zeros_like(scores_non)])
        scores = np.concatenate([scores_members, scores_non])
        auc = float(roc_auc_score(labels, scores))

        eps_total, eps_train, eps_repair = _parse_eps(path)
        repaired_flag = np.nan
        if "dp_diffusion_dpcr" in str(path).lower():
            repaired_flag = path.name.lower() == "synth.csv"
        synth_kind = "synth_unrepaired" if path.name.lower() == "synth_unrepaired.csv" else "synth"
        run_dir_key = _normalize_dir_key(str(path.parent))
        syn_path_key = _normalize_path(str(path))
        debug_payload = None
        if debug:
            debug_payload = {
                "dataset": dataset,
                "run_dir": run_dir_key,
                "syn_path": syn_path_key,
                "members_shape": list(enc_train.shape),
                "nonmembers_shape": list(enc_test.shape),
                "synth_shape": list(enc_synth.shape),
                "mean_dist_members": float(np.mean(d_members)) if len(d_members) else None,
                "mean_dist_nonmembers": float(np.mean(d_non)) if len(d_non) else None,
                "auc": float(auc),
            }
        sanity_payload = None
        if want_sanity_payload:
            sanity_payload = {
                "dataset": dataset,
                "enc_train": enc_train,
                "enc_test": enc_test,
                "enc_synth": enc_synth,
                "scores": scores,
                "labels": labels,
            }
        return {
            "run_dir": run_dir_key,
            "syn_path": syn_path_key,
            "dataset": dataset,
            "method": _parse_method(path),
            "seed": _parse_seed(path),
            "epsilon": eps_total,
            "epsilon_train": eps_train,
            "epsilon_repair": eps_repair,
            "knn_mia_auc_raw": auc,
            "knn_mia_auc": auc,
            "auc_flipped": False,
            "k": int(k),
            "n_members": int(len(enc_train)),
            "n_nonmembers": int(len(enc_test)),
            "repaired_flag": repaired_flag,
            "synth_kind": synth_kind,
            "status": "OK",
            "fail_code": "",
            "error_msg": "",
            "generated_at": generated_at,
            "run_dir_key": run_dir_key,
            "syn_path_key": syn_path_key,
        }, debug_payload, sanity_payload
    except Exception as e:
        run_dir_key = _normalize_dir_key(str(path.parent))
        syn_path_key = _normalize_path(str(path))
        return {
            "run_dir": run_dir_key,
            "syn_path": syn_path_key,
            "dataset": _infer_dataset(path),
            "method": _parse_method(path),
            "seed": _parse_seed(path),
            "epsilon": _parse_eps(path)[0],
            "epsilon_train": _parse_eps(path)[1],
            "epsilon_repair": _parse_eps(path)[2],
            "knn_mia_auc_raw": np.nan,
            "knn_mia_auc": np.nan,
            "auc_flipped": np.nan,
            "k": k,
            "n_members": np.nan,
            "n_nonmembers": np.nan,
            "repaired_flag": np.nan,
            "synth_kind": "synth_unrepaired" if path.name.lower() == "synth_unrepaired.csv" else "synth",
            "status": "FAIL",
            "fail_code": type(e).__name__,
            "error_msg": str(e)[:200],
            "generated_at": generated_at,
            "run_dir_key": run_dir_key,
            "syn_path_key": syn_path_key,
        }, None, None


def _sanity_tests(dataset: str, enc_train: np.ndarray, enc_test: np.ndarray, enc_synth: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> Dict:
    report = {"dataset": dataset}
    # S1: shuffled labels
    shuffled = labels.copy()
    RNG.shuffle(shuffled)
    auc_shuffled = float(roc_auc_score(shuffled, scores))
    report["auc_shuffled"] = auc_shuffled
    if auc_shuffled > 0.8:
        print(f"[ALERT] KNN-MIA sanity S1 failed for {dataset}: auc_shuffled={auc_shuffled:.4f}")
        raise SystemExit(1)

    # S2: random synth (noise with same standardized scale)
    synth_rand = RNG.normal(0.0, 1.0, size=enc_train.shape)
    scores_members_r, _ = _knn_scores(enc_train, synth_rand, k=1)
    scores_non_r, _ = _knn_scores(enc_test, synth_rand, k=1)
    labels_r = np.concatenate([np.ones_like(scores_members_r), np.zeros_like(scores_non_r)])
    scores_r = np.concatenate([scores_members_r, scores_non_r])
    auc_random = float(roc_auc_score(labels_r, scores_r))
    report["auc_random"] = auc_random
    if auc_random > 0.8:
        print(f"[ALERT] KNN-MIA sanity S2 failed for {dataset}: auc_random={auc_random:.4f}")
        raise SystemExit(1)

    # S3: self-match check
    def _same_sample(a: np.ndarray, b: np.ndarray) -> bool:
        if a.shape[1] != b.shape[1]:
            return False
        n = min(5, a.shape[0], b.shape[0])
        if n <= 0:
            return False
        return np.allclose(a[:n], b[:n])

    if _same_sample(enc_synth, enc_train) or _same_sample(enc_synth, enc_test):
        print(f"[ALERT] KNN-MIA sanity S3 failed for {dataset}: synth matches real sample")
        raise SystemExit(1)
    report["self_match"] = False
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--sanity", type=int, default=1)
    parser.add_argument("--max_debug_runs", type=int, default=3)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)

    outputs_dir = Path(args.outputs_dir)
    out_dir = outputs_dir / "knn_mia"
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_files = []
    synth_by_dir = {}
    for p in outputs_dir.rglob("synth.csv"):
        if "paper_package" in p.parts:
            continue
        synth_by_dir[p.parent] = p
    for p in outputs_dir.rglob("synth_unrepaired.csv"):
        if "paper_package" in p.parts:
            continue
        if p.parent in synth_by_dir:
            continue
        synth_by_dir[p.parent] = p
    synth_files = list(synth_by_dir.values())

    rows = []
    ok = 0
    fail = 0
    generated_at = datetime.utcnow().isoformat()
    debug_count = 0
    sanity_done = set()
    high_auc_streak = 0
    sanity_report = {}
    for path in synth_files:
        debug = debug_count < args.max_debug_runs
        want_sanity_payload = bool(args.sanity)
        row, dbg, sanity_payload = compute_knn_mia_for_one(
            path, args.k, generated_at, args.sanity, debug, want_sanity_payload
        )
        rows.append(row)
        if row.get("status") == "OK":
            ok += 1
        else:
            fail += 1
        if dbg:
            debug_count += 1
            print(
                f"[KNN] run={dbg['run_dir']} members={tuple(dbg['members_shape'])} "
                f"nonmembers={tuple(dbg['nonmembers_shape'])} synth={tuple(dbg['synth_shape'])} "
                f"auc={dbg['auc']:.6f}"
            )
            print(
                f"[KNN] mean_dist_members={dbg['mean_dist_members']:.6f} "
                f"mean_dist_nonmembers={dbg['mean_dist_nonmembers']:.6f}"
            )

        auc_val = row.get("knn_mia_auc")
        if isinstance(auc_val, (int, float)) and auc_val > 0.9999:
            high_auc_streak += 1
        else:
            high_auc_streak = 0
        if high_auc_streak > 5:
            raise SystemExit(1)

        dataset = row.get("dataset")
        if (
            args.sanity
            and dataset
            and dataset not in sanity_done
            and row.get("status") == "OK"
            and sanity_payload is not None
        ):
            sanity_report[dataset] = _sanity_tests(
                dataset,
                sanity_payload["enc_train"],
                sanity_payload["enc_test"],
                sanity_payload["enc_synth"],
                sanity_payload["scores"],
                sanity_payload["labels"],
            )
            sanity_done.add(dataset)

    df_out = pd.DataFrame(rows)
    preferred_cols = [
        "run_dir",
        "syn_path",
        "dataset",
        "method",
        "epsilon",
        "seed",
        "repaired_flag",
        "synth_kind",
        "knn_mia_auc_raw",
        "knn_mia_auc",
        "auc_flipped",
        "status",
        "fail_code",
        "generated_at",
        "run_dir_key",
        "syn_path_key",
    ]
    existing = [c for c in preferred_cols if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in existing]
    df_out = df_out[existing + remaining]
    csv_path_root = outputs_dir / "knn_mia_results.csv"
    json_path_root = outputs_dir / "knn_mia_results.json"
    csv_path_legacy = out_dir / "knn_mia_results.csv"
    json_path_legacy = out_dir / "knn_mia_results.json"
    df_out.to_csv(csv_path_root, index=False)
    df_out.to_json(json_path_root, orient="records", indent=2)
    df_out.to_csv(csv_path_legacy, index=False)
    df_out.to_json(json_path_legacy, orient="records", indent=2)
    if args.sanity and not sanity_report and ok > 0:
        print("[WARN] sanity enabled but no sanity_report entries were recorded.")
    if sanity_report:
        sanity_path = outputs_dir / "knn_mia_sanity.json"
        sanity_path.write_text(json.dumps(sanity_report, indent=2), encoding="utf-8")

    non_nan = int(df_out["knn_mia_auc"].notna().sum()) if "knn_mia_auc" in df_out.columns else 0
    print(f"[OK] Wrote {csv_path_root}")
    print(f"[OK] Wrote {json_path_root}")
    print(f"[SUMMARY] total={len(df_out)} OK={ok} FAIL={fail} knn_mia_auc_non_nan={non_nan}")


if __name__ == "__main__":
    main()
