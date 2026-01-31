import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime

class ComprehensiveMIAEvaluator:
    """
    Comprehensive MIA evaluation with corrected worst-case reporting.
    Implements hard gate: lower epsilon must never give worse privacy than higher epsilon.
    """
    
    def __init__(self, n_seeds=5):
        self.n_seeds = n_seeds
        self.attackers = {
            'LR': LogisticRegression(random_state=42, max_iter=1000),
            'RF': RandomForestClassifier(random_state=42, n_estimators=100),
            'MLP': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(64,)),
            'SVM': SVC(random_state=42, probability=True, kernel='rbf')
        }
        self.results = {}
    
    def compute_worst_case_metrics(self, y_true, y_scores):
        """
        Compute worst-case MIA metrics with corrected reporting.
        """
        auc = roc_auc_score(y_true, y_scores)
        
        # Worst-case AUC: max(AUC, 1-AUC) - handles scoring direction issues
        worst_case_auc = max(auc, 1 - auc)
        
        # Privacy advantage: max(|AUC - 0.5|) - distance from random guessing
        advantage = abs(auc - 0.5)
        
        return {
            'raw_auc': float(auc),
            'worst_case_auc': float(worst_case_auc),
            'advantage': float(advantage)
        }
    
    def run_sanity_check(self, X_member, X_non_member, n_trials=3):
        """
        Sanity check: shuffle membership labels should give AUC ≈ 0.5
        """
        sanity_results = []
        
        for trial in range(n_trials):
            # Combine data
            X_combined = np.vstack([X_member, X_non_member])
            y_true = np.hstack([np.ones(len(X_member)), np.zeros(len(X_non_member))])
            
            # Shuffle labels (sanity check)
            np.random.seed(trial)
            y_shuffled = np.random.permutation(y_true)
            
            # Train attacker on shuffled labels
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_shuffled, test_size=0.3, random_state=trial
            )
            
            attacker = LogisticRegression(random_state=trial, max_iter=1000)
            attacker.fit(X_train, y_train)
            y_scores = attacker.predict_proba(X_test)[:, 1]
            
            sanity_auc = roc_auc_score(y_test, y_scores)
            sanity_results.append(sanity_auc)
        
        return {
            'mean_sanity_auc': float(np.mean(sanity_results)),
            'std_sanity_auc': float(np.std(sanity_results)),
            'sanity_check_passed': bool(abs(np.mean(sanity_results) - 0.5) < 0.1)
        }
    
    def evaluate_single_setting(self, X_member, X_non_member, setting_name):
        """
        Evaluate MIA for a single setting across all attackers and seeds.
        """
        print(f"[MIA] Evaluating {setting_name}...")
        
        # Combine data and create labels
        X_combined = np.vstack([X_member, X_non_member])
        y_true = np.hstack([np.ones(len(X_member)), np.zeros(len(X_non_member))])
        
        attacker_results = {}
        
        # Run sanity check first
        sanity = self.run_sanity_check(X_member, X_non_member)
        
        # Evaluate each attacker across multiple seeds
        for attacker_name, attacker_base in self.attackers.items():
            seed_results = []
            
            for seed in range(self.n_seeds):
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_combined, y_true, test_size=0.3, random_state=seed
                    )
                    
                    # Clone and train attacker
                    if attacker_name == 'LR':
                        attacker = LogisticRegression(random_state=seed, max_iter=1000)
                    elif attacker_name == 'RF':
                        attacker = RandomForestClassifier(random_state=seed, n_estimators=100)
                    elif attacker_name == 'MLP':
                        attacker = MLPClassifier(random_state=seed, max_iter=500, hidden_layer_sizes=(64,))
                    elif attacker_name == 'SVM':
                        attacker = SVC(random_state=seed, probability=True, kernel='rbf')
                    
                    attacker.fit(X_train, y_train)
                    y_scores = attacker.predict_proba(X_test)[:, 1]
                    
                    # Compute metrics
                    metrics = self.compute_worst_case_metrics(y_test, y_scores)
                    seed_results.append(metrics)
                    
                except Exception as e:
                    print(f"[WARNING] {attacker_name} seed {seed} failed: {e}")
                    continue
            
            if seed_results:
                # Aggregate across seeds
                attacker_results[attacker_name] = {
                    'mean_worst_case_auc': float(np.mean([r['worst_case_auc'] for r in seed_results])),
                    'std_worst_case_auc': float(np.std([r['worst_case_auc'] for r in seed_results])),
                    'mean_advantage': float(np.mean([r['advantage'] for r in seed_results])),
                    'std_advantage': float(np.std([r['advantage'] for r in seed_results])),
                    'raw_aucs': [float(r['raw_auc']) for r in seed_results]
                }
        
        # Find worst-case across all attackers
        if attacker_results:
            worst_case_aucs = [result['mean_worst_case_auc'] for result in attacker_results.values()]
            advantages = [result['mean_advantage'] for result in attacker_results.values()]
            
            overall_worst_case = float(max(worst_case_aucs))
            overall_advantage = float(max(advantages))
            worst_attacker = max(attacker_results.keys(), 
                               key=lambda k: attacker_results[k]['mean_worst_case_auc'])
        else:
            overall_worst_case = 0.5
            overall_advantage = 0.0
            worst_attacker = 'None'
        
        result = {
            'setting': setting_name,
            'sanity_check': sanity,
            'attacker_results': attacker_results,
            'overall_worst_case_auc': overall_worst_case,
            'overall_advantage': overall_advantage,
            'worst_attacker': worst_attacker,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[MIA] {setting_name}: Worst-case AUC = {overall_worst_case:.3f}, Advantage = {overall_advantage:.3f}")
        
        return result
    
    def run_comprehensive_evaluation(self):
        """
        Run MIA evaluation on available DP synthetic files.
        """
        datasets = ['A', 'B', 'C']
        
        # Map available files to settings
        available_settings = {
            'A': {
                'dp_ctgan': ['A_dp_ctgan.npy'],
                'dp_diffusion': ['A_dp_eps1.0.npy', 'A_dp_eps5.0.npy']
            },
            'B': {
                'dp_ctgan': ['B_dp_ctgan.npy'], 
                'dp_diffusion': ['B_dp_eps1.0.npy', 'B_dp_eps5.0.npy']
            },
            'C': {
                'dp_ctgan': ['C_dp_ctgan.npy'],
                'dp_diffusion': ['C_dp_eps0.1.npy']
            }
        }
        
        all_results = {}
        
        for dataset in datasets:
            print(f"\n=== Dataset {dataset} ===")
            
            # Load real data for membership
            try:
                real_data = np.load(f"data/{dataset}_train.npy")
                print(f"Loaded real data: {real_data.shape}")
            except:
                print(f"[ERROR] Cannot load real data for dataset {dataset}")
                continue
            
            dataset_results = {}
            
            for method, files in available_settings[dataset].items():
                for synth_file in files:
                    if not os.path.exists(f"outputs/{synth_file}"):
                        print(f"[SKIP] outputs/{synth_file} not found")
                        continue
                    
                    # Extract epsilon from filename
                    if 'eps' in synth_file:
                        eps_str = synth_file.split('eps')[1].split('.npy')[0]
                        setting_name = f"{dataset}_{method}_eps{eps_str}"
                    else:
                        setting_name = f"{dataset}_{method}_default"
                    
                    try:
                        synth_data = np.load(f"outputs/{synth_file}")
                        print(f"Loaded synthetic data: {synth_data.shape}")
                        
                        X_member = real_data
                        X_non_member = synth_data
                        
                        # Handle dimension mismatch by using common features
                        if X_member.shape[1] != X_non_member.shape[1]:
                            min_features = min(X_member.shape[1], X_non_member.shape[1])
                            X_member = X_member[:, :min_features]
                            X_non_member = X_non_member[:, :min_features]
                            print(f"[INFO] Adjusted to {min_features} common features")
                        
                        # Ensure same size for fair comparison
                        min_size = min(len(X_member), len(X_non_member))
                        X_member = X_member[:min_size]
                        X_non_member = X_non_member[:min_size]
                        
                        # Run evaluation
                        result = self.evaluate_single_setting(X_member, X_non_member, setting_name)
                        dataset_results[setting_name] = result
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to evaluate {setting_name}: {e}")
                        continue
            
            all_results[dataset] = dataset_results
        
        self.results = all_results
        return all_results
    
    def validate_monotonic_privacy(self):
        """
        Hard gate: Validate that lower epsilon never gives worse privacy than higher epsilon.
        """
        print("\n=== Monotonic Privacy Validation ===")
        
        validation_results = {
            'passed': True,
            'violations': [],
            'summary': {}
        }
        
        # Check available epsilon pairs
        epsilon_pairs = {
            'A_dp_diffusion': [(1.0, 5.0)],  # eps1.0 vs eps5.0
            'B_dp_diffusion': [(1.0, 5.0)],  # eps1.0 vs eps5.0
        }
        
        for dataset in ['A', 'B', 'C']:
            if dataset not in self.results:
                continue
                
            for method_key, pairs in epsilon_pairs.items():
                if not method_key.startswith(dataset):
                    continue
                    
                for eps_low, eps_high in pairs:
                    setting_low = f"{dataset}_dp_diffusion_eps{eps_low}"
                    setting_high = f"{dataset}_dp_diffusion_eps{eps_high}"
                    
                    if setting_low in self.results[dataset] and setting_high in self.results[dataset]:
                        auc_low = self.results[dataset][setting_low]['overall_worst_case_auc']
                        auc_high = self.results[dataset][setting_high]['overall_worst_case_auc']
                        
                        # Lower epsilon should give better privacy (lower worst-case AUC)
                        if auc_low > auc_high + 0.02:  # Allow small tolerance
                            violation = {
                                'dataset': dataset,
                                'method': 'dp_diffusion',
                                'eps_low': float(eps_low),
                                'eps_high': float(eps_high),
                                'auc_low': float(auc_low),
                                'auc_high': float(auc_high),
                                'violation_magnitude': float(auc_low - auc_high)
                            }
                            validation_results['violations'].append(violation)
                            validation_results['passed'] = False
                            
                            print(f"[VIOLATION] {dataset}_dp_diffusion: eps={eps_low} (AUC={auc_low:.3f}) worse than eps={eps_high} (AUC={auc_high:.3f})")
                        else:
                            print(f"[OK] {dataset}_dp_diffusion: eps={eps_low} (AUC={auc_low:.3f}) <= eps={eps_high} (AUC={auc_high:.3f})")
                        
                        validation_results['summary'][f"{dataset}_dp_diffusion"] = {
                            'monotonic': bool(auc_low <= auc_high + 0.02),
                            'epsilon_aucs': {float(eps_low): float(auc_low), float(eps_high): float(auc_high)}
                        }
        
        if validation_results['passed']:
            print(f"\n[PASSED] All settings show monotonic privacy trends")
        else:
            print(f"\n[FAILED] {len(validation_results['violations'])} monotonicity violations found")
        
        return validation_results
    
    def save_results(self, filepath="comprehensive_mia_results.json"):
        """Save all results to JSON file."""
        output = {
            'evaluation_results': self.results,
            'validation': self.validate_monotonic_privacy(),
            'metadata': {
                'n_seeds': self.n_seeds,
                'attackers': list(self.attackers.keys()),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return output

def main():
    """Run comprehensive MIA evaluation."""
    print("=== Comprehensive MIA Evaluation with Worst-Case Reporting ===")
    
    evaluator = ComprehensiveMIAEvaluator(n_seeds=5)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Validate monotonic privacy
    validation = evaluator.validate_monotonic_privacy()
    
    # Save results
    output = evaluator.save_results("comprehensive_mia_results.json")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Datasets evaluated: {len(results)}")
    print(f"Total settings: {sum(len(dataset_results) for dataset_results in results.values())}")
    print(f"Monotonic validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    
    if not validation['passed']:
        print(f"Violations: {len(validation['violations'])}")
        for violation in validation['violations']:
            print(f"  - {violation['dataset']}_{violation['method']}: eps {violation['eps_low']} > eps {violation['eps_high']}")
    
    return output

if __name__ == "__main__":
    main()