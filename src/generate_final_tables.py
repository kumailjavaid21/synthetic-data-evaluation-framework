"""Generate final comparison tables for Q1 paper."""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
})

def collect_all_results():
    """Collect results from all datasets and methods."""
    
    datasets = ['A', 'B', 'C']
    all_results = []
    
    for dataset in datasets:
        # Load comprehensive evaluation results
        eval_path = f"reports/metrics/comprehensive_evaluation_{dataset}.csv"
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            all_results.append(df)
        
        # Load DP results if available
        dp_path = f"reports/metrics/dp_diffusion_evaluation.csv"
        if os.path.exists(dp_path):
            dp_df = pd.read_csv(dp_path)
            dp_df = dp_df[dp_df['Dataset'].str.contains(dataset)]
            if not dp_df.empty:
                all_results.append(dp_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        print("⚠️ No evaluation results found")
        return pd.DataFrame()

def generate_fidelity_utility_table(df):
    """Generate Table A: Fidelity & Utility per model × dataset."""
    
    if df.empty:
        return
    
    # Select key columns
    cols = ['Dataset', 'Method', 'Epsilon', 'KS_mean', 'Correlation_diff', 'NN_mean']
    table_a = df[cols].copy()
    
    # Round values
    numeric_cols = ['KS_mean', 'Correlation_diff', 'NN_mean']
    table_a[numeric_cols] = table_a[numeric_cols].round(3)
    
    # Sort by dataset then method
    table_a = table_a.sort_values(['Dataset', 'Method'])
    
    # Save as CSV
    os.makedirs("reports/final_tables", exist_ok=True)
    table_a.to_csv("reports/final_tables/table_a_fidelity_utility.csv", index=False)
    
    # Generate LaTeX table
    latex_table = table_a.to_latex(index=False, float_format="%.3f")
    with open("reports/final_tables/table_a_fidelity_utility.tex", "w") as f:
        f.write(latex_table)
    
    print("✅ Table A (Fidelity & Utility) saved")
    print(table_a.to_string(index=False))

def generate_privacy_table(df):
    """Generate Table B: Privacy metrics per model × dataset."""
    
    if df.empty:
        return
    
    # Privacy columns (if available)
    privacy_cols = ['Dataset', 'Method', 'Epsilon']
    available_privacy = []
    
    for col in ['NN_mean', 'NN_std', 'NN_min']:
        if col in df.columns:
            available_privacy.append(col)
    
    if available_privacy:
        table_b = df[privacy_cols + available_privacy].copy()
        
        # Round values
        table_b[available_privacy] = table_b[available_privacy].round(3)
        
        # Sort
        table_b = table_b.sort_values(['Dataset', 'Method'])
        
        # Save
        table_b.to_csv("reports/final_tables/table_b_privacy.csv", index=False)
        
        # LaTeX
        latex_table = table_b.to_latex(index=False, float_format="%.3f")
        with open("reports/final_tables/table_b_privacy.tex", "w") as f:
            f.write(latex_table)
        
        print("✅ Table B (Privacy) saved")
        print(table_b.to_string(index=False))

def generate_privacy_utility_curves(df):
    """Generate Figure A: Privacy-utility tradeoff curves."""
    
    if df.empty:
        return
    
    # Filter DP methods only
    dp_methods = df[df['Epsilon'] != 'inf'].copy()
    
    if dp_methods.empty:
        print("⚠️ No DP methods found for tradeoff curves")
        return
    
    # Convert epsilon to numeric
    dp_methods['Epsilon'] = pd.to_numeric(dp_methods['Epsilon'], errors='coerce')
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets = ['A', 'B', 'C']
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = dp_methods[dp_methods['Dataset'].str.contains(dataset)]
        
        if not subset.empty:
            # Plot each DP method
            for method in subset['Method'].unique():
                method_data = subset[subset['Method'] == method].sort_values('Epsilon')
                if len(method_data) > 1:
                    ax.plot(method_data['Epsilon'], method_data['KS_mean'], 
                           'o-', label=method, markersize=8, linewidth=2)
        
        ax.set_xlabel('Privacy Budget (ε)', fontweight='bold')
        ax.set_ylabel('KS Mean (↓ better utility)', fontweight='bold')
        ax.set_title(f'Dataset {dataset}', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("reports/publication_plots/figure_a_privacy_utility_curves.pdf")
    plt.close()
    
    print("✅ Figure A (Privacy-Utility Curves) saved")

def generate_method_comparison_plot(df):
    """Generate method comparison bar plot."""
    
    if df.empty:
        return
    
    # Select non-DP methods for main comparison
    non_dp = df[df['Epsilon'] == 'inf'].copy()
    
    if non_dp.empty:
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['KS_mean', 'Correlation_diff', 'NN_mean']
    titles = ['KS Statistic (↓ better)', 'Correlation Diff (↓ better)', 'NN Distance (↑ better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Pivot data
        pivot = non_dp.pivot(index='Dataset', columns='Method', values=metric)
        
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Dataset', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/publication_plots/method_comparison_barplot.pdf")
    plt.close()
    
    print("✅ Method comparison plot saved")

def main():
    """Generate all final tables and figures."""
    
    print("📊 Generating final tables and figures for Q1 paper...")
    
    # Collect all results
    df = collect_all_results()
    
    if df.empty:
        print("❌ No results found. Run experiments first.")
        return
    
    print(f"📈 Found {len(df)} result rows across {df['Dataset'].nunique()} datasets")
    
    # Generate tables
    generate_fidelity_utility_table(df)
    generate_privacy_table(df)
    
    # Generate figures
    generate_privacy_utility_curves(df)
    generate_method_comparison_plot(df)
    
    print("\n🎉 All final tables and figures generated!")
    print("📁 Check reports/final_tables/ and reports/publication_plots/")

if __name__ == "__main__":
    main()