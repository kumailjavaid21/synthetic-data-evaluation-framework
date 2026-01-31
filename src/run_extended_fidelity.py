import os, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.metrics_bjet import metric_kld_jsd_wd, metric_chisq, tstr_auc

os.makedirs("reports/metrics", exist_ok=True)

# --------- configure your pairs here ----------
# Real test / Real train / Synthetic (train-size) / (Optional) labels
PAIRS = [
    {
        "name": "A_StudentsPerformance",
        "real_train": "data/A_Xtr.npy",
        "real_test":  "data/A_Xte.npy",
        "synth":      "outputs/A_diffusion.npy",        # change to A_sdv_gc.npy to evaluate SDV
        "real_ytr":   "data/A_ytr.npy",                 # optional; if missing, TSTR is skipped
        "real_yte":   "data/A_yte.npy",                 # optional
    },
    {
        "name": "B_StudentInfo",
        "real_train": "data/B_Xtr.npy",
        "real_test":  "data/B_Xte.npy",
        "synth":      "outputs/B_diffusion_latent_repaired.npy",
        "real_ytr":   "data/B_ytr.npy",
        "real_yte":   "data/B_yte.npy",
    },
    {
        "name": "C_StudentMat",
        "real_train": "data/C_Xtr.npy",
        "real_test":  "data/C_Xte.npy",
        "synth":      "outputs/C_diffusion.npy",
        "real_ytr":   "data/C_ytr.npy",
        "real_yte":   "data/C_yte.npy",
    },
]

def safe_load(path):
    if path is None or not os.path.exists(path):
        return None
    return np.load(path)

rows = []
for cfg in PAIRS:
    name = cfg["name"]
    Xtr = np.load(cfg["real_train"])
    Xte = np.load(cfg["real_test"])
    Xs  = np.load(cfg["synth"])

    print(f"\n[{name}] Extended utility metrics on: real_test {Xte.shape} vs synth {Xs.shape}")

    # KLD/JSD/WD + Chi-square
    u1 = metric_kld_jsd_wd(Xte, Xs, bins=32)
    u2 = metric_chisq(Xte, Xs, bins=20)

    # TSTR if labels exist
    ytr = safe_load(cfg.get("real_ytr"))
    yte = safe_load(cfg.get("real_yte"))
    if ytr is not None and yte is not None:
        tstr = tstr_auc(Xtr, ytr, Xte, yte, Xs)
    else:
        tstr = {"TSTR_metric": "NA (no labels provided)"}

    row = {"Dataset": name, **u1, **u2, **tstr}
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = "reports/metrics/extended_fidelity.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ Saved → {out_csv}")
print(df)
