import os, json, pandas as pd
from src.evaluate_synthetic import evaluate_pair

os.makedirs("reports/metrics", exist_ok=True)

pairs = [
    ("A_StudentsPerformance", "data/A_Xte.npy", "outputs/A_sdv_gc.npy"),
    ("B_StudentInfo", "data/B_Xte.npy", "outputs/B_sdv_gc.npy"),
    ("C_StudentMat", "data/C_Xte.npy", "outputs/C_sdv_gc.npy"),
]
# pairs = [
#     ("A_StudentsPerformance", "data/A_Xte.npy", "outputs/A_diffusion.npy"),
#     ("B_StudentInfo",         "data/B_Xte.npy", "outputs/B_diffusion.npy"),
#     ("C_StudentMat",          "data/C_Xte.npy", "outputs/C_diffusion.npy"),
# ]


results = []
for name, real_path, synth_path in pairs:
    print(f"Evaluating {name} ...")
    result = evaluate_pair(real_path, synth_path, name)
    results.append(result)

df = pd.DataFrame(results)
df.to_csv("reports/metrics/sdv_evaluation.csv", index=False)
print("\n>>> Metrics saved to reports/metrics/sdv_evaluation.csv")
print(df)
