import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.privacy_bjet import (
    metric_k_anonymity,
    metric_delta_presence,
    metric_identifiability,
    metric_membership_inference,
)

os.makedirs("reports/metrics", exist_ok=True)

PAIRS = [
    ("A_StudentsPerformance", "data/A_Xte.npy", "outputs/A_diffusion.npy"),
    ("B_StudentInfo",         "data/B_Xte.npy", "outputs/B_diffusion_latent.npy"),
    ("C_StudentMat",          "data/C_Xte.npy", "outputs/C_diffusion.npy"),
]

results = []
for name, real_path, synth_path in PAIRS:
    print(f"\nEvaluating privacy metrics for {name} ...")
    R = np.load(real_path)
    S = np.load(synth_path)

    k_res   = metric_k_anonymity(R, S)
    d_res   = metric_delta_presence(R, S)
    i_res   = metric_identifiability(R, S)
    mia_res = metric_membership_inference(R, S)

    row = {"Dataset": name, **k_res, **d_res, **i_res, **mia_res}
    results.append(row)

df = pd.DataFrame(results)
out_path = "reports/metrics/extended_privacy.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Privacy metrics saved → {out_path}")
print(df)
