import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


class StandardMIAEvaluator:
    """Standard MIA evaluation protocol - fixes inconsistencies."""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def evaluate_mia(self, real_data, synthetic_data):
        """Standard MIA protocol: 0.5 = good privacy, >0.5 = leak."""
        # Create balanced dataset: 50% real (label=1), 50% synthetic (label=0)
        X = np.vstack([real_data, synthetic_data])
        y = np.hstack([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        
        # Cross-validation for robust evaluation
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        aucs = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train attacker
            attacker = LogisticRegression(random_state=self.random_state)
            attacker.fit(X_train, y_train)
            
            # Get attack scores
            scores = attacker.predict_proba(X_test)[:, 1]  # Probability of being real
            
            # Standard ROC-AUC (no inversion)
            auc = roc_auc_score(y_test, scores)
            aucs.append(auc)
        
        return {
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'aucs': aucs,
            'interpretation': self._interpret_auc(np.mean(aucs))
        }
    
    def _interpret_auc(self, auc):
        """Correct AUC interpretation."""
        if 0.45 <= auc <= 0.55:
            return "Good privacy (random guessing)"
        elif auc > 0.6:
            return "Privacy leak detected"
        elif auc < 0.4:
            return "Possible protocol error or label inversion"
        else:
            return "Moderate privacy"


def fix_mia_inconsistency():
    """Fix MIA AUC inconsistencies by using single standard protocol."""
    evaluator = StandardMIAEvaluator()
    
    # Example usage - replace with actual data
    real_data = np.random.randn(1000, 10)
    synthetic_data = np.random.randn(1000, 10) + 0.1  # Slightly different
    
    result = evaluator.evaluate_mia(real_data, synthetic_data)
    print(f"MIA AUC: {result['mean_auc']:.3f} ± {result['std_auc']:.3f}")
    print(f"Interpretation: {result['interpretation']}")
    
    return result