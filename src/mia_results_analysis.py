"""
Comprehensive MIA Evaluation Results Summary
============================================

CRITICAL FINDINGS:
- Most DP synthetic data shows PERFECT VULNERABILITY (AUC = 1.0)
- This indicates DP mechanisms are NOT working properly
- Only some Logistic Regression attacks show reasonable privacy

EVALUATION SCOPE:
- Datasets: A, B, C
- Methods: DP-CTGAN, DP-Diffusion  
- Attackers: LR, RF, MLP, SVM (5 seeds each)
- Metrics: Worst-case AUC = max(AUC, 1-AUC), Advantage = |AUC - 0.5|

KEY RESULTS:
"""

import json
import pandas as pd

def analyze_mia_results():
    """Analyze and summarize MIA evaluation results."""
    
    with open('comprehensive_mia_results.json', 'r') as f:
        results = json.load(f)
    
    print("=== COMPREHENSIVE MIA EVALUATION SUMMARY ===\n")
    
    # Summary statistics
    total_settings = 0
    perfect_attacks = 0
    reasonable_privacy = 0
    
    summary_data = []
    
    for dataset, dataset_results in results['evaluation_results'].items():
        print(f"Dataset {dataset}:")
        print("-" * 40)
        
        for setting, result in dataset_results.items():
            total_settings += 1
            
            # Extract key metrics
            overall_auc = result['overall_worst_case_auc']
            overall_advantage = result['overall_advantage']
            worst_attacker = result['worst_attacker']
            sanity_passed = result['sanity_check']['sanity_check_passed']
            
            # Count attack success
            if overall_auc >= 0.95:
                perfect_attacks += 1
                privacy_level = "BROKEN"
            elif overall_auc >= 0.7:
                privacy_level = "POOR"
            elif overall_auc >= 0.6:
                privacy_level = "WEAK"
            else:
                reasonable_privacy += 1
                privacy_level = "REASONABLE"
            
            print(f"  {setting}:")
            print(f"    Worst-case AUC: {overall_auc:.3f}")
            print(f"    Advantage: {overall_advantage:.3f}")
            print(f"    Worst attacker: {worst_attacker}")
            print(f"    Privacy level: {privacy_level}")
            print(f"    Sanity check: {'PASSED' if sanity_passed else 'FAILED'}")
            
            # Detailed attacker breakdown
            print(f"    Attacker AUCs:")
            for attacker, attacker_result in result['attacker_results'].items():
                auc = attacker_result['mean_worst_case_auc']
                print(f"      {attacker}: {auc:.3f}")
            print()
            
            # Store for summary table
            summary_data.append({
                'Dataset': dataset,
                'Setting': setting,
                'Worst_Case_AUC': overall_auc,
                'Advantage': overall_advantage,
                'Worst_Attacker': worst_attacker,
                'Privacy_Level': privacy_level,
                'Sanity_Passed': sanity_passed
            })
    
    # Overall summary
    print("=== OVERALL SUMMARY ===")
    print(f"Total settings evaluated: {total_settings}")
    print(f"Perfect attacks (AUC >= 0.95): {perfect_attacks} ({perfect_attacks/total_settings*100:.1f}%)")
    print(f"Reasonable privacy (AUC < 0.6): {reasonable_privacy} ({reasonable_privacy/total_settings*100:.1f}%)")
    
    # Monotonic validation
    validation = results['validation']
    print(f"\nMonotonic privacy validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    if validation['violations']:
        print(f"Violations found: {len(validation['violations'])}")
        for violation in validation['violations']:
            print(f"  - {violation['dataset']}_{violation['method']}: eps {violation['eps_low']} worse than eps {violation['eps_high']}")
    
    # Critical issues
    print("\n=== CRITICAL ISSUES IDENTIFIED ===")
    
    issue_count = 0
    
    # Issue 1: Perfect attacks
    if perfect_attacks > 0:
        issue_count += 1
        print(f"{issue_count}. PERFECT ATTACKS: {perfect_attacks}/{total_settings} settings show AUC = 1.0")
        print("   - This indicates DP mechanisms are completely broken")
        print("   - Synthetic data is perfectly distinguishable from real data")
    
    # Issue 2: Random Forest dominance
    rf_dominance = sum(1 for data in summary_data if data['Worst_Attacker'] == 'RF')
    if rf_dominance > total_settings * 0.5:
        issue_count += 1
        print(f"{issue_count}. RF DOMINANCE: Random Forest is worst attacker in {rf_dominance}/{total_settings} cases")
        print("   - Suggests synthetic data has structural patterns RF can exploit")
    
    # Issue 3: Epsilon effectiveness
    epsilon_pairs = [
        ('A_dp_diffusion_eps1.0', 'A_dp_diffusion_eps5.0'),
        ('B_dp_diffusion_eps1.0', 'B_dp_diffusion_eps5.0')
    ]
    
    epsilon_working = False
    for low_eps, high_eps in epsilon_pairs:
        for data in summary_data:
            if data['Setting'] == low_eps:
                low_auc = data['Worst_Case_AUC']
            elif data['Setting'] == high_eps:
                high_auc = data['Worst_Case_AUC']
        
        if 'low_auc' in locals() and 'high_auc' in locals():
            if abs(low_auc - high_auc) > 0.05:  # Meaningful difference
                epsilon_working = True
    
    if not epsilon_working:
        issue_count += 1
        print(f"{issue_count}. EPSILON INEFFECTIVE: No meaningful privacy difference between epsilon values")
        print("   - DP training may not be applying noise correctly")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    print("1. URGENT: Investigate DP training implementation")
    print("   - Verify noise is actually being added during training")
    print("   - Check gradient clipping and noise calibration")
    print("   - Validate epsilon accounting is correct")
    
    print("2. Debug Random Forest attacks")
    print("   - RF achieving perfect separation suggests structural issues")
    print("   - May indicate synthetic data has artificial patterns")
    
    print("3. Re-run with corrected DP implementation")
    print("   - Fix DP training mechanisms first")
    print("   - Then re-evaluate with same MIA framework")
    
    # Save summary table
    df = pd.DataFrame(summary_data)
    df.to_csv('mia_evaluation_summary.csv', index=False)
    print(f"\nDetailed results saved to: mia_evaluation_summary.csv")
    
    return {
        'total_settings': total_settings,
        'perfect_attacks': perfect_attacks,
        'reasonable_privacy': reasonable_privacy,
        'validation_passed': validation['passed'],
        'critical_issues': issue_count
    }

if __name__ == "__main__":
    summary = analyze_mia_results()
    
    print(f"\n=== FINAL STATUS ===")
    if summary['perfect_attacks'] > 0:
        print("[FAILED] DP mechanisms are broken (perfect attacks possible)")
    elif summary['reasonable_privacy'] == summary['total_settings']:
        print("[PASSED] All settings show reasonable privacy")
    else:
        print("[MIXED] Some privacy issues remain")
    
    print(f"Monotonic validation: {'[PASSED]' if summary['validation_passed'] else '[FAILED]'}")
    print(f"Critical issues found: {summary['critical_issues']}")