"""Generate high-quality EDA plots from scratch for publication."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,
})

def create_output_dir():
    """Create publication plots directory."""
    out_dir = Path("reports/publication_plots")
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir

def plot_distributions(dataset_name, data_path, title):
    """Generate distribution plots for a dataset."""
    out_dir = create_output_dir()
    
    try:
        data = np.load(data_path)
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"⚠️ Data file not found: {data_path}")
        return
    
    # Select subset of features for visualization (max 12)
    n_features = min(12, df.shape[1])
    selected_cols = df.columns[:n_features]
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_cols):
        ax = axes[i]
        
        # Plot histogram with KDE
        ax.hist(df[col], bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE line
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(df[col].dropna())
            x_range = np.linspace(df[col].min(), df[col].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_title(f'Feature {i+1}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title} - Feature Distributions', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_distributions.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/{dataset_name}_distributions.pdf")

def plot_correlation(dataset_name, data_path, title):
    """Generate correlation heatmap for a dataset."""
    out_dir = create_output_dir()
    
    try:
        data = np.load(data_path)
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"⚠️ Data file not found: {data_path}")
        return
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=False,  # No annotations for cleaner look
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                ax=ax)
    
    ax.set_title(f'{title} - Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('Features', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_correlation.pdf")
    plt.close()
    print(f"✅ Saved → {out_dir}/{dataset_name}_correlation.pdf")

def main():
    """Generate all EDA plots from scratch."""
    print("🎨 Generating high-quality EDA plots from scratch...")
    
    # Dataset configurations
    datasets = [
        ("A_StudentsPerformance", "data/A_Xtr.npy", "Dataset A: Students Performance"),
        ("B_StudentInfo", "data/B_Xtr.npy", "Dataset B: Student Info (OULAD)"),
        ("C_StudentMat", "data/C_Xtr.npy", "Dataset C: Student Math Grades")
    ]
    
    for dataset_name, data_path, title in datasets:
        print(f"\n📊 Processing {title}...")
        plot_distributions(dataset_name, data_path, title)
        plot_correlation(dataset_name, data_path, title)
    
    print("\n✅ All EDA plots generated in reports/publication_plots/")
    print("📄 Files ready for publication:")
    for dataset_name, _, _ in datasets:
        print(f"   - {dataset_name}_distributions.pdf")
        print(f"   - {dataset_name}_correlation.pdf")

if __name__ == "__main__":
    main()