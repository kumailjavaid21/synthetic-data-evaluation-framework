import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class RobustMIAEvaluator:
    """Robust MIA evaluation with conservative worst-case reporting."""
    
    def __init__(self, n_splits=3, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.attackers = {
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state)
        }
    
    def _validate_data_construction(self, real_data, synthetic_data):
        """Validate MIA dataset construction."""
        # Check for exact duplicates between real and synthetic
        real_set = set(map(tuple, real_data))
        synth_set = set(map(tuple, synthetic_data))
        overlap = len(real_set.intersection(synth_set))
        
        if overlap > 0:
            raise ValueError(f"Data leak: {overlap} exact duplicates between real and synthetic")
        
        # Check dimensions match
        if real_data.shape[1] != synthetic_data.shape[1]:
            raise ValueError("Real and synthetic data must have same number of features")
        
        print(f"✅ Data validation passed: {len(real_data)} real, {len(synthetic_data)} synthetic")
    
    def _sanity_check_shuffled_labels(self, X, y):
        """Sanity check: shuffled labels should give AUC ≈ 0.5."""
        y_shuffled = np.random.permutation(y)
        
        # Quick single-fold test
        n_train = len(X) // 2
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y_shuffled[:n_train], y_shuffled[n_train:]
        
        attacker = LogisticRegression(random_state=self.random_state, max_iter=1000)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        attacker.fit(X_train_scaled, y_train)
        scores = attacker.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, scores)
        
        if not (0.4 <= auc <= 0.6):
            print(f"⚠️  Sanity check warning: Shuffled labels AUC = {auc:.3f} (expected ~0.5)")
        else:
            print(f"✅ Sanity check passed: Shuffled labels AUC = {auc:.3f}")
        
        return auc
    
    def evaluate_single_attacker(self, real_data, synthetic_data, attacker_name):
        """Evaluate single attacker with proper scoring direction."""
        self._validate_data_construction(real_data, synthetic_data)
        
        # Create balanced dataset: real=1 (member), synthetic=0 (non-member)
        X = np.vstack([real_data, synthetic_data])
        y = np.hstack([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        
        # Sanity check with shuffled labels
        sanity_auc = self._sanity_check_shuffled_labels(X, y)
        
        # Cross-validation evaluation
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        aucs = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train attacker
            attacker = self.attackers[attacker_name]
            attacker.fit(X_train_scaled, y_train)
            
            # Get membership probability (probability of being real/member)
            member_probs = attacker.predict_proba(X_test_scaled)[:, 1]
            
            # Compute AUC: y_test=1 for real data, member_probs should be high
            auc = roc_auc_score(y_test, member_probs)
            aucs.append(auc)
            
            print(f"  Fold {fold+1}: AUC = {auc:.3f}")
        
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        return {
            'attacker': attacker_name,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'aucs': aucs,
            'sanity_auc': sanity_auc
        }
    
    def evaluate_all_attackers(self, real_data, synthetic_data):
        """Evaluate all attackers and compute conservative metrics."""
        print(f"\n🔍 Evaluating MIA with {len(self.attackers)} attackers...")
        
        results = {}
        raw_aucs = []
        
        for attacker_name in self.attackers.keys():
            print(f"\n--- {attacker_name} ---")
            result = self.evaluate_single_attacker(real_data, synthetic_data, attacker_name)
            results[attacker_name] = result
            raw_aucs.append(result['mean_auc'])
        
        # Conservative worst-case reporting
        conservative_aucs = [max(auc, 1 - auc) for auc in raw_aucs]  # max(AUC, 1-AUC)
        worst_case_auc = max(conservative_aucs)
        worst_case_attacker = list(self.attackers.keys())[np.argmax(conservative_aucs)]
        
        # Privacy advantage (distance from random)
        advantages = [abs(auc - 0.5) for auc in raw_aucs]
        max_advantage = max(advantages)
        
        summary = {
            'raw_aucs': raw_aucs,
            'conservative_aucs': conservative_aucs,
            'worst_case_auc': worst_case_auc,
            'worst_case_attacker': worst_case_attacker,
            'max_advantage': max_advantage,
            'privacy_status': self._interpret_privacy(worst_case_auc),
            'individual_results': results
        }
        
        print(f"\n📊 SUMMARY:")
        print(f"Raw AUCs: {[f'{auc:.3f}' for auc in raw_aucs]}")
        print(f"Conservative AUCs: {[f'{auc:.3f}' for auc in conservative_aucs]}")
        print(f"Worst-case AUC: {worst_case_auc:.3f} ({worst_case_attacker})")
        print(f"Max advantage: {max_advantage:.3f}")
        print(f"Privacy status: {summary['privacy_status']}")
        
        return summary
    
    def _interpret_privacy(self, worst_case_auc):
        """Interpret privacy based on worst-case AUC."""
        if worst_case_auc <= 0.55:
            return "Good privacy"
        elif worst_case_auc <= 0.65:
            return "Moderate privacy"
        elif worst_case_auc <= 0.75:
            return "Privacy leak"
        else:
            return "Severe privacy leak"


def fix_mia_evaluation_final():
    """Final fix for MIA evaluation with conservative reporting."""
    evaluator = RobustMIAEvaluator()
    
    # Generate test data with known privacy properties
    np.random.seed(42)
    real_data = np.random.randn(1000, 10)
    
    # Synthetic data with varying similarity to real data
    test_cases = {
        'Good Privacy': np.random.randn(1000, 10),  # Independent
        'Moderate Privacy': real_data + np.random.randn(1000, 10) * 0.5,  # Some correlation
        'Privacy Leak': real_data + np.random.randn(1000, 10) * 0.1  # High correlation
    }
    
    results = {}
    for case_name, synthetic_data in test_cases.items():
        print(f"\n{'='*50}")
        print(f"TEST CASE: {case_name}")
        print(f"{'='*50}")
        
        result = evaluator.evaluate_all_attackers(real_data, synthetic_data)
        results[case_name] = result
    
    return results, evaluator


if __name__ == "__main__":
    results, evaluator = fix_mia_evaluation_final()
    
    # Create corrected results table
    corrected_data = []
    for case, result in results.items():
        corrected_data.append({
            'Test_Case': case,
            'Worst_Case_AUC': result['worst_case_auc'],
            'Max_Advantage': result['max_advantage'],
            'Privacy_Status': result['privacy_status'],
            'Worst_Attacker': result['worst_case_attacker']
        })
    
    df = pd.DataFrame(corrected_data)
    print(f"\n{'='*60}")
    print("CORRECTED MIA EVALUATION RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    df.to_csv('corrected_mia_results.csv', index=False)
    print(f"\n✅ Saved corrected results to 'corrected_mia_results.csv'")