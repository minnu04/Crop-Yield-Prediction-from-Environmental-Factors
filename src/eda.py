"""
Exploratory Data Analysis module.
Generates visualizations and statistical summaries of the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Print basic dataset information.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print("="*60 + "\n")


def create_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Create and save correlation heatmap for numerical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    output_dir : str
        Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/correlation_heatmap.png")


def create_feature_vs_yield_plots(df: pd.DataFrame, target: str = 'crop_yield', 
                                   output_dir: str = "outputs") -> None:
    """
    Create scatter/regression plots showing relationship between features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name
    output_dir : str
        Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col != target]
    
    # Create a subplot grid
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        ax.scatter(df[feature], df[target], alpha=0.5, s=20)
        z = np.polyfit(df[feature], df[target], 1)
        p = np.poly1d(z)
        ax.plot(df[feature].sort_values(), p(df[feature].sort_values()), 
                "r--", linewidth=2, label='Trend line')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(target, fontsize=10)
        ax.set_title(f'{feature} vs {target}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_vs_yield_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/feature_vs_yield_scatter.png")


def create_yield_by_crop_type_boxplot(df: pd.DataFrame, crop_col: str = 'crop_type',
                                       target: str = 'crop_yield', 
                                       output_dir: str = "outputs") -> None:
    """
    Create boxplot of target variable by crop type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    crop_col : str
        Categorical column name (crop type)
    target : str
        Target column name
    output_dir : str
        Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=crop_col, y=target, hue=crop_col, palette='Set2', legend=False)
    plt.title(f'{target.title()} Distribution by {crop_col.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xlabel(crop_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(target.replace("_", " ").title(), fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yield_by_crop_type_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/yield_by_crop_type_boxplot.png")


def create_yield_distribution_plot(df: pd.DataFrame, target: str = 'crop_yield',
                                    output_dir: str = "outputs") -> None:
    """
    Create distribution plot of the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name
    output_dir : str
        Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    axes[0].hist(df[target], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel(target.replace("_", " ").title(), fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Distribution of {target.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # KDE plot
    df[target].plot(kind='kde', ax=axes[1], linewidth=2, color='darkblue')
    axes[1].fill_between(axes[1].get_lines()[0].get_xdata(), 
                         axes[1].get_lines()[0].get_ydata(), alpha=0.3)
    axes[1].set_xlabel(target.replace("_", " ").title(), fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title(f'KDE Plot of {target.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yield_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/yield_distribution.png")


def perform_eda(df: pd.DataFrame, target: str = 'crop_yield', 
                crop_col: str = 'crop_type', output_dir: str = "outputs") -> None:
    """
    Perform complete exploratory data analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name
    crop_col : str
        Categorical column name
    output_dir : str
        Directory to save plots
    """
    print("\n" + "="*60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    print_dataset_info(df)
    create_correlation_heatmap(df, output_dir)
    create_feature_vs_yield_plots(df, target, output_dir)
    create_yield_by_crop_type_boxplot(df, crop_col, target, output_dir)
    create_yield_distribution_plot(df, target, output_dir)
    
    print("="*60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
