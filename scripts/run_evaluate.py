from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, encoder: OneHotEncoder):
    num = df.select_dtypes(include=[np.number])
    cat = encoder.transform(df.select_dtypes(include=["object"]))
    return np.hstack((num, cat))


def utility(real: pd.DataFrame, synth: pd.DataFrame, target: str, task: str):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    features = real.drop(columns=[target])
    encoder.fit(features.select_dtypes(include=["object"]))
    X_real = preprocess(features, encoder)
    X_synth = preprocess(synth.drop(columns=[target]), encoder)
    y = real[target]
    model = LogisticRegression(max_iter=200) if task == "classification" else Ridge()
    model.fit(X_real, y)
    return {"tstr_score_real": model.score(X_real, y), "tstr_score_synth": model.score(X_synth, synth[target])}


def privacy(real: pd.DataFrame, synth: pd.DataFrame):
    real_train, real_test = train_test_split(real, test_size=0.4, random_state=42)
    nbr = NearestNeighbors(n_neighbors=1).fit(real_train.select_dtypes(include=[np.number]))
    d_real = nbr.kneighbors(real_test.select_dtypes(include=[np.number]), return_distance=True)[0]
    if len(synth.select_dtypes(include=[np.number]).index) == 0:
        return {"privacy_advantage": 0.0}
    nbr_syn = NearestNeighbors(n_neighbors=1).fit(
        synth.select_dtypes(include=[np.number])
    )
    d_synth = nbr_syn.kneighbors(real_test.select_dtypes(include=[np.number]), return_distance=True)[0]
    return {"privacy_advantage": float(np.mean(d_synth) - np.mean(d_real))}


def fidelity(real: pd.DataFrame, synth: pd.DataFrame):
    nums = real.select_dtypes(include=[np.number]).columns
    corr_real = real[nums].corr()
    corr_synth = synth[nums].corr()
    corr_mad = float((corr_real - corr_synth).abs().mean().mean())
    ks_stats = [kstest(real[col], synth[col]).statistic for col in nums if col in synth]
    return {"corr_mad": corr_mad, "ks_mean": float(np.mean(ks_stats)) if ks_stats else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--synth", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--outdir", default="for_github/outputs")
    args = parser.parse_args()

    real = load_csv(Path(args.real))
    synth = load_csv(Path(args.synth))
    util = utility(real, synth, args.target, args.task)
    priv = privacy(real, synth)
    fid = fidelity(real, synth)

    outdir = Path(args.outdir)
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary = {**util, **priv, **fid}
    pd.DataFrame([summary]).to_csv(tables / "summary.csv", index=False)

    plt.figure(figsize=(3.5, 2.4))
    plt.bar(summary.keys(), summary.values(), color="gray")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "summary_metrics.png", dpi=150)


if __name__ == "__main__":
    main()
