import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

KEYWORDS = [
    "TSTR",
    "RTR",
    "balanced accuracy",
    "BalAcc",
    "train_test_split",
    "StratifiedKFold",
    "KFold",
    "cross_val",
    "split",
    "seed",
    "random_state",
    "LogisticRegression",
    "RandomForest",
]

DATASET_RE = re.compile(r"[/\\\\]([ABC])[/\\\\]")
DATASET_EQ_RE = re.compile(r"(?:dataset|ds)\s*[:=_-]\s*([ABC])", re.IGNORECASE)


def _scan_scripts(repo_root: Path, verbose: bool) -> Dict[str, List[str]]:
    scripts_dir = repo_root / "scripts"
    self_path = Path(__file__).resolve()
    hits: Dict[str, List[str]] = {}
    if not scripts_dir.exists():
        return hits
    for path in scripts_dir.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()
        matches: List[str] = []
        for i, line in enumerate(lines, 1):
            for kw in KEYWORDS:
                if kw.lower() in line.lower():
                    matches.append(f"{path.as_posix()}:{i}:{line.strip()}")
                    break
        if matches:
            hits[str(path)] = matches
    if verbose:
        print(f"[AUDIT] script_hits={sum(len(v) for v in hits.values())} files={len(hits)}")
    return hits


def _extract_protocol(hits: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    evidence: List[str] = []
    tts_lines: List[str] = []
    kfold_lines: List[str] = []
    seed_lines: List[str] = []

    preferred_files = []
    for path in hits.keys():
        p = Path(path)
        if p.name in {"run_tstr_and_corr_eval.py", "utility_tstr_and_corr.py"}:
            preferred_files.append(p)

    preferred = []
    for p in preferred_files:
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if any(tok in line for tok in ["train_test_split", "stratify", "random_state", "split_seed", "test_size", "Falling back to non-stratified split", "StratifiedKFold", "KFold"]):
                preferred.append(f"{p.as_posix()}:{i}:{line.strip()}")

    scan_lines = preferred if preferred else [line for lines in hits.values() for line in lines]
    for line in scan_lines:
        lower = line.lower()
        if "train_test_split" in lower:
            tts_lines.append(line)
        if "stratifiedkfold" in lower or re.search(r"\\bkfold\\b", lower):
            kfold_lines.append(line)
        if "random_state" in lower or re.search(r"\\bseed\\b", lower):
            seed_lines.append(line)

    evidence.extend(tts_lines[:8])
    evidence.extend(kfold_lines[:8])
    evidence.extend(seed_lines[:8])

    text = "\n".join(tts_lines + kfold_lines + seed_lines)
    seeds = re.findall(r"(random_state|seed)\\s*=\\s*([a-zA-Z0-9_]+)", text, flags=re.IGNORECASE)
    seed_vals = sorted({s for _, s in seeds})

    desc_parts: List[str] = []
    if tts_lines:
        test_size = re.search(r"test_size\\s*=\\s*([0-9.]+)", text)
        strat_present = any("stratify=" in l.lower() for l in scan_lines)
        fallback_present = any("falling back to non-stratified split" in l.lower() for l in scan_lines)
        rs = re.search(r"random_state\\s*=\\s*([a-zA-Z0-9_]+)", text)
        desc_parts.append("train/test holdout split")
        if test_size:
            desc_parts.append(f"test_size={test_size.group(1)}")
        if strat_present:
            if fallback_present:
                desc_parts.append("stratified when possible; fallback to non-stratified if needed")
            else:
                desc_parts.append("stratified=yes")
        else:
            desc_parts.append("stratified=no")
        if rs:
            desc_parts.append(f"random_state={rs.group(1)}")
    elif kfold_lines:
        n_splits = re.search(r"n_splits\\s*=\\s*([0-9]+)", text)
        shuffle = re.search(r"shuffle\\s*=\\s*(True|False)", text)
        rs = re.search(r"random_state\\s*=\\s*([0-9]+)", text)
        desc_parts.append("KFold CV")
        if n_splits:
            desc_parts.append(f"n_splits={n_splits.group(1)}")
        if shuffle:
            desc_parts.append(f"shuffle={shuffle.group(1)}")
        if rs:
            desc_parts.append(f"random_state={rs.group(1)}")
    else:
        desc_parts.append("split protocol not conclusively found; inferred from keyword matches")

    if seed_vals:
        desc_parts.append(f"seeds={','.join(seed_vals[:5])}{'...' if len(seed_vals) > 5 else ''}")

    description = "Split: " + ", ".join(desc_parts) + "."
    return description, evidence


def _candidate_csvs(repo_root: Path, outputs_root: Path) -> List[Path]:
    candidates: List[Path] = []
    out_rel = outputs_root.as_posix().rstrip("/")
    patterns = [
        f"{out_rel}/final_results_summary_paper_safe.csv",
        f"{out_rel}/**/final_results_summary*.csv",
        "final_results_bundle*/tables/*.csv",
        f"{out_rel}/master_results*.csv",
        "outputs*/master_results*.csv",
    ]
    for pat in patterns:
        candidates.extend(repo_root.glob(pat))
    return sorted({p for p in candidates if p.is_file()})


def _score_csv(df: pd.DataFrame) -> int:
    cols = [c.lower() for c in df.columns]
    score = 0
    if any("dataset" in c for c in cols):
        score += 5
    if any("tstr" in c for c in cols):
        score += 5
    if any(("balacc" in c) or ("balanced" in c) for c in cols):
        score += 3
    return score


def _pick_best_csv(candidates: List[Path]) -> Optional[Path]:
    best = None
    best_score = -1
    best_size = -1
    for path in candidates:
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        score = _score_csv(df)
        size = path.stat().st_size
        if score > best_score or (score == best_score and size > best_size):
            best_score = score
            best_size = size
            best = path
    return best


def _infer_dataset_from_row(row: pd.Series, cols: List[str]) -> Optional[str]:
    for c in cols:
        val = row.get(c)
        if not isinstance(val, str):
            continue
        m = DATASET_RE.search(val)
        if m:
            return m.group(1)
        m = DATASET_EQ_RE.search(val)
        if m:
            return m.group(1).upper()
    return None


def _is_truthy(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1", "y"}
    return bool(val)


def _is_falsey(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"false", "no", "0", "n"}
    return val is False or val == 0


def _find_exclusions(df: pd.DataFrame) -> Tuple[int, Dict[str, int], str, bool]:
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    dataset_col = None
    for c in cols:
        if "dataset" in c.lower():
            dataset_col = c
            break

    tstr_cols = [
        c
        for c in cols
        if "tstr" in c.lower()
        and ("bal" in c.lower() or "balanced" in c.lower() or "balacc" in c.lower())
    ]
    tstr_col = tstr_cols[0] if tstr_cols else None

    flag_cols = []
    for c in cols:
        cl = c.lower()
        if any(tok in cl for tok in ["missing_label", "label_missing", "invalid", "excluded", "skip", "status", "label_ok"]):
            flag_cols.append(c)

    excluded_mask = pd.Series(False, index=df.index)
    if tstr_col:
        excluded_mask |= df[tstr_col].isna()

    for c in flag_cols:
        cl = c.lower()
        if "label_ok" in cl:
            excluded_mask |= df[c].apply(_is_falsey)
        elif "status" in cl:
            excluded_mask |= df[c].astype(str).str.contains("fail|skip|invalid", case=False, na=False)
        else:
            excluded_mask |= df[c].apply(_is_truthy)

    excluded = df[excluded_mask].copy()
    total = int(excluded_mask.sum())

    counts = {"A": 0, "B": 0, "C": 0}
    dataset_inferable = dataset_col is not None
    if dataset_col:
        for ds in counts:
            counts[ds] = int((excluded[dataset_col].astype(str).str.upper() == ds).sum())
    else:
        path_cols = [c for c in cols if any(tok in c.lower() for tok in ["run_dir", "syn_path", "path", "config"])]
        for _, row in excluded.iterrows():
            ds = _infer_dataset_from_row(row, path_cols)
            if ds in counts:
                counts[ds] += 1
                dataset_inferable = True

    reason = "missing/invalid label"
    return total, counts, reason, dataset_inferable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outputs_root = Path(args.outputs_root)

    hits = _scan_scripts(repo_root, args.verbose)
    extra_files = [
        repo_root / "run_tstr_and_corr_eval.py",
        repo_root / "src" / "utility_tstr_and_corr.py",
    ]
    for path in extra_files:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()
        matches = []
        for i, line in enumerate(lines, 1):
            for kw in KEYWORDS:
                if kw.lower() in line.lower():
                    matches.append(f"{path.as_posix()}:{i}:{line.strip()}")
                    break
        if matches:
            hits[str(path)] = matches
    split_desc, evidence = _extract_protocol(hits)

    candidates = _candidate_csvs(repo_root, outputs_root)
    best_csv = _pick_best_csv(candidates)
    total_excluded = 0
    per_ds = {"A": 0, "B": 0, "C": 0}
    reason = "missing/invalid label"

    if best_csv is not None:
        df = pd.read_csv(best_csv)
        total_excluded, per_ds, reason, dataset_inferable = _find_exclusions(df)
        if args.verbose:
            print(f"[AUDIT] best_csv={best_csv.as_posix()} rows={len(df)}")

    if args.verbose and evidence:
        print("[AUDIT] split_evidence:")
        for line in evidence[:20]:
            print(f"  {line}")

    print(split_desc)
    if dataset_inferable:
        print(f"Excluded runs: A={per_ds['A']}, B={per_ds['B']}, C={per_ds['C']} (reason: {reason}).")
    else:
        print(f"Excluded runs (total): {total_excluded} (reason: {reason}).")


if __name__ == "__main__":
    main()
