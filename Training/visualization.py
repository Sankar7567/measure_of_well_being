import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def save_plot(filename: str, output_dirs: list) -> None:
    """
    Saves the current active matplotlib figure to multiple output directories.
    """
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {filepath}")
    plt.close()

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, output_dirs: list) -> None:
    """
    Builds a correlation matrix heatmap using Seaborn.
    """
    plt.figure(figsize=(8, 6))
    
    # Merge features and target for correlation analysis
    df_corr = X.copy()
    df_corr["HDI"] = y
    
    # Exclude encoded Country from correlation heatmap for clean presentation of numeric indicators
    if "Country" in df_corr.columns:
        df_corr_numeric = df_corr.drop(columns=["Country"])
    else:
        df_corr_numeric = df_corr

    corr_matrix = df_corr_numeric.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
    plt.title("Correlation Matrix Heatmap of Development Indicators", fontsize=14, pad=15)
    
    save_plot("correlation_heatmap.png", output_dirs)

def plot_distribution_plots(y: pd.Series, output_dirs: list) -> None:
    """
    Plots the probability density and distribution of the Human Development Index (HDI).
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(y, kde=True, bins=20, color="#17a2b8", edgecolor="white", alpha=0.7)
    plt.title("Human Development Index (HDI) Distribution", fontsize=14, pad=15)
    plt.xlabel("HDI score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    save_plot("hdi_distribution.png", output_dirs)

def plot_strip_plots(X: pd.DataFrame, y: pd.Series, output_dirs: list) -> None:
    """
    Plots strip plots for Mean Years of Schooling and Life Expectancy vs HDI.
    Uses the first 20 rows of the dataset to avoid overcrowding.
    """
    # Create subset of first 20 rows
    df_subset = X.head(20).copy()
    df_subset["HDI"] = y.head(20)

    # 1. Mean Years of Schooling vs HDI
    plt.figure(figsize=(10, 6))
    g = sns.stripplot(x="Mean years of schooling", y="HDI", data=df_subset, jitter=True, size=8, color="#007bff", alpha=0.8)
    plt.xticks(rotation=90)
    plt.title("Mean Years of Schooling vs HDI (First 20 Countries)", fontsize=14, pad=15)
    plt.xlabel("Mean Years of Schooling", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    save_plot("schooling_vs_hdi_strip.png", output_dirs)

    # 2. Life Expectancy vs HDI
    plt.figure(figsize=(10, 6))
    g = sns.stripplot(x="Life expectancy", y="HDI", data=df_subset, jitter=True, size=8, color="#dc3545", alpha=0.8)
    plt.xticks(rotation=90)
    plt.title("Life Expectancy vs HDI (First 20 Countries)", fontsize=14, pad=15)
    plt.xlabel("Life Expectancy (years)", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    save_plot("life_expectancy_vs_hdi_strip.png", output_dirs)

def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, output_dirs: list) -> None:
    """
    Plots actual values vs predicted values with a diagonal line representing perfect prediction.
    """
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, color="#28a745", alpha=0.6, edgecolors="white", s=80, label="Predictions")
    
    # Draw reference diagonal line
    lims = [
        min(y_test.min(), y_pred.min()) - 0.05,
        max(y_test.max(), y_pred.max()) + 0.05
    ]
    plt.plot(lims, lims, color="#dc3545", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    
    plt.xlabel("Actual HDI Score", fontsize=12)
    plt.ylabel("Predicted HDI Score", fontsize=12)
    plt.title("Actual vs Predicted HDI (Linear Regression Model)", fontsize=14, pad=15)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    
    save_plot("actual_vs_predicted.png", output_dirs)

def generate_all_plots(X: pd.DataFrame, y: pd.Series, output_dirs: list) -> None:
    """
    Generates and saves all initial exploratory data visualizations.
    """
    print("Generating exploratory data visualizations...")
    plot_correlation_heatmap(X, y, output_dirs)
    plot_distribution_plots(y, output_dirs)
    plot_strip_plots(X, y, output_dirs)
    print("EDA Visualizations successfully generated.")
