import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utility_tstr_and_corr import resolve_label_col


def _pick_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _label_counts(df: pd.DataFrame, label_col: str) -> dict:
    return df[label_col].value_counts(dropna=False).to_dict()


def main() -> int:
    train_path = Path("data") / "processed" / "B" / "train.csv"
    test_path = Path("data") / "processed" / "B" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("[AUDIT] Missing processed B train/test CSVs.")
        return 1

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    label_col, _, _ = resolve_label_col("B", train_df)
    if label_col is None or label_col not in train_df.columns:
        print("[AUDIT] label_col not found in train.csv")
        return 1

    print(f"[AUDIT] label_col={label_col}")
    print(f"[AUDIT] train shape={train_df.shape} test shape={test_df.shape}")
    print(f"[AUDIT] train label counts={_label_counts(train_df, label_col)}")
    print(f"[AUDIT] test label counts={_label_counts(test_df, label_col)}")

    train_features = [c for c in train_df.columns if c != label_col]

    synth_candidates = {
        "dp_diffusion": _pick_first(list((Path("outputs") / "B" / "dp_diffusion").rglob("synth.csv"))),
        "dp_diffusion_dpcr": _pick_first(list((Path("outputs") / "B" / "dp_diffusion_dpcr").rglob("synth.csv"))),
        "sdv": _pick_first(list((Path("outputs") / "B" / "baselines_sdv").rglob("synth.csv"))),
    }

    for key, syn_path in synth_candidates.items():
        if syn_path is None:
            print(f"[AUDIT] synth path for {key}: NOT FOUND")
            continue
        syn_df = pd.read_csv(syn_path)
        syn_label, _, _ = resolve_label_col("B", syn_df)
        print(f"[AUDIT] synth path={syn_path}")
        print(f"[AUDIT] synth shape={syn_df.shape}")
        if syn_label is None or syn_label not in syn_df.columns:
            print("[AUDIT] synth label missing")
            return 1
        counts = _label_counts(syn_df, syn_label)
        print(f"[AUDIT] synth label counts={counts}")
        if syn_df[syn_label].nunique(dropna=False) <= 1:
            print("[AUDIT] synth label is constant")
            return 1

        missing_cols = sorted([c for c in train_features if c not in syn_df.columns])
        extra_cols = sorted([c for c in syn_df.columns if c not in train_df.columns])
        print(f"[AUDIT] missing_cols_in_synth={missing_cols}")
        print(f"[AUDIT] extra_cols_in_synth={extra_cols}")

        intersect = [c for c in train_features if c in syn_df.columns]
        if len(intersect) < 10:
            print("[AUDIT] feature intersection < 10")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
