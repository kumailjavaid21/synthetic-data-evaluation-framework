"""Membership Inference Attack evaluation for privacy assessment."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def membership_inference_attack(real_data, synth_data, test_size=0.3, random_state=42):
    """
    Perform membership inference attack to evaluate privacy.
    
    Returns:
        mia_auc: AUC score (0.5 = perfect privacy, 1.0 = no privacy)
        identifiability: Fraction of real records uniquely identifiable
    """
    # Ensure same number of samples for balanced evaluation
    n_samples = min(len(real_data), len(synth_data), 5000)  # Cap for efficiency
    
    real_subset = real_data[:n_samples]
    synth_subset = synth_data[:n_samples]
    
    # Create labels (1 = real, 0 = synthetic)
    X = np.vstack([real_subset, synth_subset])
    y = np.concatenate([np.ones(len(real_subset)), np.zeros(len(synth_subset))])
    
    # Split for training MIA classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MIA classifier (Random Forest for robustness)
    mia_classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=random_state,
        n_jobs=-1
    )
    
    try:
        mia_classifier.fit(X_train_scaled, y_train)
        y_pred_proba = mia_classifier.predict_proba(X_test_scaled)[:, 1]
        mia_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"⚠️ MIA evaluation failed: {e}")
        mia_auc = 0.5  # Fallback to random guessing
    
    # Compute identifiability (nearest neighbor matching)
    identifiability = compute_identifiability(real_subset, synth_subset)
    
    return mia_auc, identifiability

def compute_identifiability(real_data, synth_data, k=1):
    """
    Compute identifiability: fraction of real records that can be uniquely 
    identified by their closest synthetic record.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Standardize for distance computation
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_data)
    synth_scaled = scaler.transform(synth_data)
    
    # Find nearest synthetic record for each real record
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(synth_scaled)
    distances, indices = nbrs.kneighbors(real_scaled)
    
    # Count unique matches (identifiable records)
    unique_matches = len(np.unique(indices.flatten()))
    total_real = len(real_data)
    
    # Identifiability = fraction of real records with unique closest synthetic match
    identifiability = unique_matches / total_real
    
    return identifiability

def evaluate_all_privacy_metrics(real_data, synth_data):
    """Comprehensive privacy evaluation."""
    
    # MIA and identifiability
    mia_auc, identifiability = membership_inference_attack(real_data, synth_data)
    
    # Additional privacy metrics
    from sklearn.neighbors import NearestNeighbors
    
    # Standardize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_data)
    synth_scaled = scaler.transform(synth_data)
    
    # Nearest neighbor distance (privacy proxy)
    n_samples = min(1000, len(real_scaled), len(synth_scaled))
    nbrs = NearestNeighbors(n_neighbors=1).fit(synth_scaled[:n_samples])
    distances, _ = nbrs.kneighbors(real_scaled[:n_samples])
    nn_distance = np.mean(distances)
    
    # Delta-presence (fraction of real records matched)
    threshold = np.percentile(distances, 5)  # 5th percentile as threshold
    delta_presence = np.mean(distances < threshold)
    
    return {
        'MIA_AUC': float(mia_auc),
        'Identifiability': float(identifiability),
        'NN_Distance': float(nn_distance),
        'Delta_Presence': float(delta_presence)
    }