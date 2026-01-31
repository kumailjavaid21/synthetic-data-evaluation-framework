import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "final_results_bundle_ABC" / "tables"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _check_dpcr_pairs(path: Path) -> None:
    if not path.exists():
        _fail(f"Missing {path}")
    df = pd.read_csv(path)
    if df.empty:
        _fail("dpcr_pairs.csv is empty")
    if "dpcr_used_file" not in df.columns:
        _fail("dpcr_pairs missing dpcr_used_file")
    used = df["dpcr_used_file"].fillna("").astype(str)
    pct = used.str.lower().str.endswith("synth.csv").mean() * 100
    if pct < 100.0:
        _fail(f"dpcr_used_file not 100% synth.csv (pct={pct:.1f})")
    if "delta_utility" not in df.columns:
        _fail("dpcr_pairs missing delta_utility")
    if df["delta_utility"].isna().any():
        _fail("dpcr_pairs has NaN delta_utility")
    key_cols = ["dataset", "seed", "epsilon_train", "epsilon_repair_dpcr"]
    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        _fail(f"dpcr_pairs missing key cols: {missing_keys}")
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes:
        _fail(f"dpcr_pairs has duplicate keys: {dupes}")
    print(f"[OK] dpcr_pairs rows={len(df)}")


def _check_dpcr_summary(path: Path) -> None:
    if not path.exists():
        _fail(f"Missing {path}")
    df = pd.read_csv(path)
    if df.empty:
        _fail("dpcr_effect_summary_compact.csv is empty")
    datasets = set(df["dataset"].dropna().astype(str))
    if datasets != {"A", "B", "C"}:
        _fail(f"dpcr_effect_summary_compact datasets={sorted(datasets)}")
    ci_cols = [c for c in df.columns if c.endswith("_ci95_low") or c.endswith("_ci95_high")]
    if not ci_cols:
        _fail("dpcr_effect_summary_compact missing CI columns")
    if "n_pairs" not in df.columns:
        _fail("dpcr_effect_summary_compact missing n_pairs")
    for _, row in df.iterrows():
        if row["n_pairs"] < 2:
            for c in ci_cols:
                if pd.notna(row.get(c)):
                    _fail("CI values present when n_pairs < 2")
    print(f"[OK] dpcr_effect_summary_compact rows={len(df)}")


def _check_money_tables() -> None:
    for ds in ["A", "B", "C"]:
        path = TABLES_DIR / f"money_table_{ds}.csv"
        if not path.exists():
            _fail(f"Missing {path}")
        df = pd.read_csv(path)
        if df.empty:
            _fail(f"money_table_{ds} is empty")
        if "delta_vs_baseline_privacy" not in df.columns or "delta_vs_baseline_utility" not in df.columns:
            _fail(f"money_table_{ds} missing delta_vs_baseline columns")
        if df["delta_vs_baseline_privacy"].notna().sum() <= 0:
            _fail(f"money_table_{ds} delta_vs_baseline_privacy all NaN")
        if df["delta_vs_baseline_utility"].notna().sum() <= 0:
            _fail(f"money_table_{ds} delta_vs_baseline_utility all NaN")
        if "n_seeds" not in df.columns:
            _fail(f"money_table_{ds} missing n_seeds")
        if (pd.to_numeric(df["n_seeds"], errors="coerce") < 1).any():
            _fail(f"money_table_{ds} has n_seeds < 1")
        print(f"[OK] money_table_{ds} rows={len(df)}")


def main() -> None:
    dpcr_pairs = TABLES_DIR / "dpcr_pairs.csv"
    dpcr_summary = TABLES_DIR / "dpcr_effect_summary_compact.csv"
    _check_dpcr_pairs(dpcr_pairs)
    _check_dpcr_summary(dpcr_summary)
    _check_money_tables()
    print("[OK] audit_tables_publication_ready passed")


if __name__ == "__main__":
    main()
