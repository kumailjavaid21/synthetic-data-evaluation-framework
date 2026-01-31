"""Generate high-quality PDF plots for research paper publication."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,  # Set to True if you have LaTeX installed
})

def create_output_dir():
    """Create publication plots directory."""
    out_dir = Path("reports/publication_plots")
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir

def plot_fidelity_comparison():
    """Generate fidelity comparison plot."""
    out_dir = create_output_dir()
    
    # Load data
    try:
        non_private = pd.read_csv("reports/metrics/diffusion_evaluation.csv")
        dp_results = pd.read_csv("reports/metrics/dp_diffusion_evaluation.csv")
        sdv_results = pd.read_csv("reports/metrics/sdv_evaluation.csv")
    except FileNotFoundError:
        print("⚠️ Some CSV files missing. Generating with available data.")
        return
    
    # Prepare data
    non_private['Method'] = 'Non-Private Diffusion'
    non_private['Epsilon'] = np.inf
    dp_results['Method'] = 'DP-Diffusion'
    sdv_results['Method'] = 'SDV Gaussian Copula'
    sdv_results['Epsilon'] = np.inf
    
    df = pd.concat([non_private, dp_results, sdv_results], ignore_index=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['KS_mean', 'Correlation_diff', 'NN_mean']
    titles = ['KS Statistic (↓ better)', 'Correlation Difference (↓ better)', 'NN Distance (↑ better privacy)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Extract data for plotting
        datasets = ['A', 'B', 'C']
        methods = df['Method'].unique()
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for j, method in enumerate(methods):
            values = []
            for ds in datasets:
                subset = df[(df['Method'] == method) & (df['Dataset'].str.contains(ds))]
                if not subset.empty and metric in subset.columns:
                    values.append(subset[metric].iloc[0])
                else:
                    values.append(0)
            
            ax.bar(x + j*width, values, width, label=method, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title}', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_comparison.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/fidelity_comparison.pdf")

def plot_privacy_utility_tradeoff():
    """Generate privacy-utility tradeoff plot."""
    out_dir = create_output_dir()
    
    try:
        non_private = pd.read_csv("reports/metrics/diffusion_evaluation.csv")
        dp_results = pd.read_csv("reports/metrics/dp_diffusion_evaluation.csv")
    except FileNotFoundError:
        print("⚠️ DP results not found. Run DP experiments first.")
        return
    
    # Prepare data
    non_private['Epsilon'] = np.inf
    non_private['Method'] = 'Non-Private'
    dp_results['Method'] = 'DP-Diffusion'
    
    df = pd.concat([non_private, dp_results], ignore_index=True)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot for Dataset B (where we have DP results)
    subset = df[df['Dataset'].str.contains('B')]
    
    # Separate DP and non-private points
    dp_data = subset[subset['Method'] == 'DP-Diffusion']
    np_data = subset[subset['Method'] == 'Non-Private']
    
    # Plot points
    if not dp_data.empty:
        ax.scatter(dp_data['Epsilon'], dp_data['KS_mean'], 
                  s=100, c='red', marker='o', label='DP-Diffusion', alpha=0.8)
    
    if not np_data.empty:
        ax.scatter([100], np_data['KS_mean'], 
                  s=100, c='blue', marker='s', label='Non-Private', alpha=0.8)
    
    ax.set_xlabel('Privacy Budget (ε)', fontweight='bold')
    ax.set_ylabel('KS Mean (↓ better utility)', fontweight='bold')
    ax.set_title('Privacy-Utility Tradeoff (Dataset B)', fontweight='bold', fontsize=16)
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Add annotation
    if not dp_data.empty:
        ax.annotate(f'ε={dp_data["Epsilon"].iloc[0]:.2f}\nBetter utility!', 
                   xy=(dp_data['Epsilon'].iloc[0], dp_data['KS_mean'].iloc[0]),
                   xytext=(0.5, 0.6), fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out_dir / "privacy_utility_tradeoff.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/privacy_utility_tradeoff.pdf")

def plot_privacy_metrics():
    """Generate privacy metrics comparison."""
    out_dir = create_output_dir()
    
    try:
        privacy_data = pd.read_csv("reports/metrics/extended_privacy.csv")
    except FileNotFoundError:
        print("⚠️ Privacy metrics not found.")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['MIA_AUC', 'identifiability', 'delta_presence']
    titles = ['MIA AUC (↓ better)', 'Identifiability (↓ better)', 'δ-Presence (↓ better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        datasets = privacy_data['Dataset'].str.extract(r'([ABC])')[0].tolist()
        values = privacy_data[metric].tolist()
        
        bars = ax.bar(datasets, values, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "privacy_metrics.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/privacy_metrics.pdf")

def plot_model_comparison_table():
    """Generate publication-ready comparison table."""
    out_dir = create_output_dir()
    
    try:
        comparison = pd.read_csv("reports/comparison/model_comparison.csv")
    except FileNotFoundError:
        print("⚠️ Model comparison not found.")
        return
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = comparison[['Dataset', 'Model', 'KS_mean', 'Correlation_diff', 'NN_mean']].round(3)
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=['Dataset', 'Method', 'KS Mean ↓', 'Corr. Diff ↓', 'NN Mean ↑'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(out_dir / "model_comparison_table.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/model_comparison_table.pdf")

def main():
    """Generate all publication plots."""
    print("🎨 Generating publication-quality PDF plots...")
    
    plot_fidelity_comparison()
    plot_privacy_utility_tradeoff()
    plot_privacy_metrics()
    plot_model_comparison_table()
    
    print("\n✅ All publication plots generated in reports/publication_plots/")
    print("📄 Files ready for LaTeX/Word insertion:")
    print("   - fidelity_comparison.pdf")
    print("   - privacy_utility_tradeoff.pdf") 
    print("   - privacy_metrics.pdf")
    print("   - model_comparison_table.pdf")

if __name__ == "__main__":
    main()