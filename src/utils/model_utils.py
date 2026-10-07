"""Shared utilities for ML model evaluation and visualization."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def calculate_fill_factor(x, y):
    """Calculate fill factor as the trapezoidal integral area."""
    return np.trapezoid(y, x)


def plot_shap_summary(shap_values, X, feature_names, title, plot_type, file_path):
    """Plot SHAP summary (bar or dot) for tree-based model interpretation."""
    plt.rcParams['font.family'] = 'Arial'
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type=plot_type, cmap='coolwarm', show=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Adjust scatter point size
    ax = plt.gca()
    for collection in ax.collections:
        collection.set_sizes([20])

    plt.savefig(file_path, dpi=600)
    plt.close()


def plot_bar_chart(categories, values, labels, file_path):
    """Plot a bar chart comparing model performance metrics."""
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=categories, y=values, palette='pastel')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='center', fontsize=10,
                    color='black', xytext=(0, 5), textcoords='offset points')

    # Add MAE and MSE text
    max_value = max(values)
    for i, (mae, mse) in enumerate(zip(*labels)):
        plt.text(i, max_value + 0.05, f'MAE: {mae:.2f}', ha='center', fontsize=9)
        plt.text(i, max_value + 0.10, f'MSE: {mse:.2f}', ha='center', fontsize=9)

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R2 Score', fontsize=12)
    plt.ylim(0, max_value + 0.15)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_heatmap(data, feature_names, file_path):
    """Plot a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm',
                     square=True, linewidths=.5, vmin=-1, vmax=1,
                     xticklabels=feature_names, yticklabels=feature_names)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def save_shap_values(shap_values, feature_names, output_dir, prefix):
    """Save SHAP values to CSV."""
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    save_path = os.path.join(output_dir, 'data', f'{prefix}_shap_values.csv')
    shap_df.to_csv(save_path, index=False)
    print(f"SHAP values saved to: {save_path}")


def save_model_performance(models, metrics, output_dir, cv=True):
    """Save model performance metrics to CSV."""
    suffix = 'cv_performance' if cv else 'performance'
    performance_df = pd.DataFrame({
        'Model': models,
        'R2_CV_Avg' if cv else 'R2': metrics['R2'],
        'MSE_CV_Avg' if cv else 'MSE': metrics['MSE'],
        'MAE_CV_Avg' if cv else 'MAE': metrics['MAE']
    })
    save_path = os.path.join(output_dir, 'data', f'model_{suffix}.csv')
    performance_df.to_csv(save_path, index=False)
    print(f"Model performance saved to: {save_path}")


def save_correlation_matrix(corr_matrix, feature_names, output_dir):
    """Save correlation matrix to CSV."""
    corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
    save_path = os.path.join(output_dir, 'data', 'feature_correlation.csv')
    corr_df.to_csv(save_path)
    print(f"Correlation matrix saved to: {save_path}")


def save_feature_importance(importances, feature_names, output_dir, prefix):
    """Save feature importance values to CSV."""
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    save_path = os.path.join(output_dir, 'data', f'{prefix}_feature_importance.csv')
    importance_df.to_csv(save_path, index=False)
    print(f"Feature importance saved to: {save_path}")
