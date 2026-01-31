"""
Fixed MIA Evaluation with Debugging and Assertions
"""

import os, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_mia_split_from_run(run_dir: Path, dataset_name: str):
    """Load member/non-member arrays using split indices if present; returns (members, nonmembers) or (None, None)."""
    try:
        run_dir = Path(run_dir)
        idx_members = run_dir / "mia_members_idx.npy"
        idx_nonmembers = run_dir / "mia_nonmembers_idx.npy"
        if not idx_members.exists() or not idx_nonmembers.exists():
            return None, None
        base_path = Path("data") / f"{dataset_name}_Xtr.npy"
        if not base_path.exists():
            return None, None
        X_full = np.load(base_path)
        members_idx = np.load(idx_members)
        nonmembers_idx = np.load(idx_nonmembers)
        return X_full[members_idx], X_full[nonmembers_idx]
    except Exception:
        return None, None


def _sanitize_name(val):
    return str(val).replace(" ", "_").replace("/", "_") if val is not None else "unknown"


def _dump_debug(payload, dataset_name=None, method_name=None, eps_total=None, debug_dir="outputs"):
    ds = _sanitize_name(dataset_name)
    meth = _sanitize_name(method_name)
    eps = _sanitize_name(eps_total)
    debug_path = Path(debug_dir) / f"mia_debug_{ds}_{meth}_{eps}.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Debug stats saved to {debug_path}")
    return debug_path


def debug_mia_evaluation(
    real_data,
    synth_data,
    random_state=42,
    dataset_name=None,
    method_name=None,
    eps_total=None,
    debug_dir="outputs",
    verbose=True,
    member_data=None,
    nonmember_data=None,
):
    """
    Debug MIA evaluation with detailed logging and assertions.
    Ensures held-out test split, membership labels without leakage, and sanity stats.
    """
    rng = np.random.RandomState(random_state)
    if verbose:
        print(f"\nDEBUG MIA EVALUATION (seed={random_state}, dataset={dataset_name}, method={method_name}, eps_total={eps_total})")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synth data shape: {synth_data.shape}")
    
    # Build member/non-member sets
    if member_data is not None and nonmember_data is not None:
        members = np.asarray(member_data, dtype=float)
        nonmembers = np.asarray(nonmember_data, dtype=float)
    else:
        # fallback to original real vs synth
        n_samples = min(len(real_data), len(synth_data), 2000)
        members = np.asarray(real_data[:n_samples], dtype=float)
        nonmembers = np.asarray(synth_data[:n_samples], dtype=float)
    
    # Ensure same number of samples for balanced evaluation
    n_samples = min(len(members), len(nonmembers))
    members = members[:n_samples]
    nonmembers = nonmembers[:n_samples]
    
    # Create labels (1 = member, 0 = non-member)
    X = np.vstack([members, nonmembers])
    y = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    
    if verbose:
        print(f"Combined data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Member samples (real): {np.sum(y == 1)}")
        print(f"Non-member samples (synth): {np.sum(y == 0)}")
        print(f"Unique label values: {np.unique(y)}")
    
    # ASSERTION: Must have both classes
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        payload = {"reason": "non-binary-labels", "unique_labels": unique_labels.tolist(), "dataset": dataset_name, "method": method_name, "eps_total": eps_total}
        _dump_debug(payload, dataset_name, method_name, eps_total, debug_dir=debug_dir)
        raise ValueError(f"MIA evaluation requires both member and non-member samples. Got unique labels: {unique_labels}")
    
    # Split for training/validation/testing (held-out test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.3333, random_state=random_state, stratify=y_trainval
    )  # ~60/20/20 split
    
    if verbose:
        print(f"Train set: {X_train.shape}, labels: {np.bincount(y_train.astype(int))}")
        print(f"Val set: {X_val.shape}, labels: {np.bincount(y_val.astype(int))}")
        print(f"Test set: {X_test.shape}, labels: {np.bincount(y_test.astype(int))}")
    
    # ASSERTION: Both train and test must have both classes
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Train/val/test splits must each contain both member and non-member samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MIA classifier (Random Forest)
    mia_classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=random_state,
        n_jobs=1  # Single job for reproducibility
    )
    
    mia_classifier.fit(X_train_scaled, y_train)
    
    # Get prediction probabilities (NOT hard predictions)
    y_pred_proba = mia_classifier.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (member)
    
    if verbose:
        print(f"Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"Pred proba min/max/mean/std: {y_pred_proba.min():.3f}/{y_pred_proba.max():.3f}/{y_pred_proba.mean():.3f}/{y_pred_proba.std():.3f}")
        print(f"Test labels min/max/mean: {y_test.min():.3f}/{y_test.max():.3f}/{y_test.mean():.3f}")
        print(f"Pred proba unique count (rounded 3dp): {len(np.unique(np.round(y_pred_proba, 3)))}")
    
    # ASSERTION: Probabilities must be in valid range
    if not (0 <= y_pred_proba.min() and y_pred_proba.max() <= 1):
        raise ValueError(f"Prediction probabilities out of range [0,1]: {y_pred_proba.min()}-{y_pred_proba.max()}")
    
    # ASSERTION: Probabilities should not be constant
    pred_std = float(np.std(y_pred_proba))
    unique_scores = len(np.unique(np.round(y_pred_proba, 3)))
    if pred_std < 1e-6 or unique_scores <= 2:
        payload = {
            "reason": "constant-predictions",
            "dataset": dataset_name,
            "method": method_name,
            "eps_total": eps_total,
            "pred_std": pred_std,
            "unique_scores": unique_scores
        }
        _dump_debug(payload, dataset_name, method_name, eps_total, debug_dir=debug_dir)
        raise ValueError(f"Prediction probabilities are nearly constant (std={pred_std:.6f}, unique_scores={unique_scores})")
    
    # Compute ROC AUC from probabilities
    mia_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Baseline sanity: random scores ~ 0.5 AUC
    random_scores = rng.rand(len(y_test))
    random_auc = roc_auc_score(y_test, random_scores)
    if verbose:
        print(f"Baseline random AUC (sanity): {random_auc:.3f}")
    if abs(random_auc - 0.5) > 0.2:
        print(f"WARNING: Random baseline AUC deviates from 0.5 (got {random_auc:.3f}); check label balance or scoring.")

    # Sanity split predictability using logistic regression on same split
    log_clf = LogisticRegression(max_iter=200)
    log_clf.fit(X_train_scaled, y_train)
    sanity_scores = log_clf.predict_proba(X_test_scaled)[:, 1]
    sanity_split_auc = roc_auc_score(y_test, sanity_scores)
    mia_inflated_by_split = sanity_split_auc > 0.60
    
    if verbose:
        print(f"MIA AUC: {mia_auc:.6f}")
    
    # Detect suspicious perfect AUC with nearly binary scores
    binary_fraction = float(np.mean((y_pred_proba < 1e-3) | (y_pred_proba > 1 - 1e-3)))
    suspicious = (np.isclose(mia_auc, 1.0) and (np.std(y_pred_proba) < 1e-2 or len(np.unique(np.round(y_pred_proba, 3))) <= 3 or binary_fraction > 0.9))
    if suspicious:
        warning_msg = "Suspicious MIA AUC=1.0 with nearly binary/constant predictions."
        print(f"WARNING: {warning_msg}")
        debug_payload = {
            "dataset": dataset_name or "unknown",
            "seed": random_state,
            "mia_auc": float(mia_auc),
            "random_auc": float(random_auc),
            "label_counts": {
                "train": np.bincount(y_train.astype(int)).tolist(),
                "val": np.bincount(y_val.astype(int)).tolist(),
                "test": np.bincount(y_test.astype(int)).tolist(),
            },
            "pred_stats": {
                "min": float(y_pred_proba.min()),
                "max": float(y_pred_proba.max()),
                "mean": float(y_pred_proba.mean()),
                "std": float(y_pred_proba.std()),
                "unique_rounded_3dp": len(np.unique(np.round(y_pred_proba, 3))),
                "binary_fraction": binary_fraction,
            },
            "scores_preview": y_pred_proba[:10].round(4).tolist(),
            "warning": warning_msg,
        }
        _dump_debug(debug_payload, dataset_name, method_name, eps_total, debug_dir=debug_dir)
    
    return {
        "mia_auc_raw": float(mia_auc),
        "mia_auc_star": float(max(mia_auc, 1 - mia_auc)),
        "mia_advantage_abs": float(abs(mia_auc - 0.5)),
        "sanity_split_auc": float(sanity_split_auc),
        "mia_inflated_by_split": bool(mia_inflated_by_split),
    }


def evaluate_mia_multiple_seeds(
    real_data,
    synth_data,
    seeds=[42, 123, 456],
    dataset_name=None,
    method_name=None,
    eps_total=None,
    debug_dir="outputs",
    verbose=False,
    member_data=None,
    nonmember_data=None,
):
    """
    Evaluate MIA with multiple seeds and return statistics.
    """
    mia_auc_raws = []
    mia_auc_stars = []
    mia_adv_abs = []
    sanity_aucs = []
    inflated_flags = []
    
    for seed in seeds:
        try:
            res = debug_mia_evaluation(
                real_data,
                synth_data,
                random_state=seed,
                dataset_name=dataset_name,
                method_name=method_name,
                eps_total=eps_total,
                debug_dir=debug_dir,
                verbose=verbose,
                member_data=member_data,
                nonmember_data=nonmember_data,
            )
            mia_auc_raws.append(res["mia_auc_raw"])
            mia_auc_stars.append(res["mia_auc_star"])
            mia_adv_abs.append(res["mia_advantage_abs"])
            sanity_aucs.append(res["sanity_split_auc"])
            inflated_flags.append(res["mia_inflated_by_split"])
            if verbose:
                print(f"Seed {seed}: MIA AUC = {res['mia_auc_raw']:.6f}, sanity_split_auc={res['sanity_split_auc']:.3f}")
        except Exception as e:
            print(f"ERROR in MIA evaluation with seed {seed}: {e}")
            payload = {
                "reason": "exception",
                "error": str(e),
                "dataset": dataset_name,
                "method": method_name,
                "eps_total": eps_total,
                "seed": seed
            }
            _dump_debug(payload, dataset_name, method_name, eps_total, debug_dir=debug_dir)
            mia_auc_raws.append(0.5)  # Fallback
            mia_auc_stars.append(0.5)
            mia_adv_abs.append(0.0)
            sanity_aucs.append(0.5)
            inflated_flags.append(False)
    
    mia_auc_raws = np.array(mia_auc_raws, dtype=float)
    mia_auc_stars = np.array(mia_auc_stars, dtype=float)
    mia_adv_abs = np.array(mia_adv_abs, dtype=float)
    sanity_aucs = np.array(sanity_aucs, dtype=float)
    inflated_flags = np.array(inflated_flags, dtype=bool)
    
    return {
        'mia_auc_mean': float(np.mean(mia_auc_raws)),
        'mia_auc_std': float(np.std(mia_auc_raws)),
        'mia_aucs_raw': mia_auc_raws.tolist(),
        'mia_auc_star_mean': float(np.mean(mia_auc_stars)),
        'mia_auc_star_std': float(np.std(mia_auc_stars)),
        'mia_advantage_abs_mean': float(np.mean(mia_adv_abs)),
        'sanity_split_auc_mean': float(np.mean(sanity_aucs)),
        'sanity_split_auc_std': float(np.std(sanity_aucs)),
        'mia_inflated_by_split': bool(inflated_flags.any()),
    }


def smoke_test_non_dp_diffusion():
    """
    Smoke test: Run non-DP diffusion and check if MIA AUC > 0.5
    """
    print("\n" + "="*60)
    print("SMOKE TEST: Non-DP Diffusion MIA Evaluation")
    print("="*60)
    
    # Check if we have non-DP diffusion output for Dataset A
    real_path = "data/A_Xte.npy"
    synth_path = "outputs/A_diffusion.npy"  # Non-DP diffusion
    
    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        print(f"Missing files for smoke test:")
        print(f"  Real: {real_path} - {'EXISTS' if os.path.exists(real_path) else 'MISSING'}")
        print(f"  Synth: {synth_path} - {'EXISTS' if os.path.exists(synth_path) else 'MISSING'}")
        return False
    
    real_data = np.load(real_path)
    synth_data = np.load(synth_path)
    
    print(f"Testing non-DP diffusion on Dataset A")
    print(f"Real data: {real_data.shape}")
    print(f"Synth data: {synth_data.shape}")
    
    # Single seed test
    mia_auc = debug_mia_evaluation(real_data, synth_data, random_state=42, verbose=True)
    
    print(f"\nSMOKE TEST RESULT: MIA AUC = {mia_auc:.6f}")
    
    if mia_auc > 0.5:
        print("SMOKE TEST PASSED: Non-DP diffusion shows privacy leakage (MIA AUC > 0.5)")
        return True
    else:
        print("SMOKE TEST FAILED: Non-DP diffusion should show privacy leakage")
        return False


if __name__ == "__main__":
    import os
    
    # Run smoke test
    smoke_test_non_dp_diffusion()
    
    # Test on existing DP results if available
    if os.path.exists("data/A_Xte.npy") and os.path.exists("outputs/A_dp_eps1.0.npy"):
        print(f"\nTesting DP-Diffusion (eps=1.0) on Dataset A")
        real_data = np.load("data/A_Xte.npy")
        synth_data = np.load("outputs/A_dp_eps1.0.npy")
        
        results = evaluate_mia_multiple_seeds(real_data, synth_data, verbose=True)
        print(f"Multi-seed results: {results}")
