"""
Visualization utilities for ML project.
Creates plots and charts to understand model performance and data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from pathlib import Path

try:
    from .config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_feature_correlations(data, target_column='price', save_path=None):
    """
    Plot correlation heatmap of features.
    
    Args:
        data (pd.DataFrame): Dataset
        target_column (str): Name of target column
        save_path (Path): Path to save the plot
    """
    logger.info("Creating correlation heatmap")
    
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlations
    correlations = numerical_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name of the model
        save_path (Path): Path to save the plot
    """
    logger.info(f"Creating predictions vs actual plot for {model_name}")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title(f'{model_name}: Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_residuals(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot residuals (prediction errors).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name of the model
        save_path (Path): Path to save the plot
    """
    logger.info(f"Creating residual plot for {model_name}")
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual scatter plot
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Price ($)', fontsize=12)
    ax1.set_ylabel('Residuals ($)', fontsize=12)
    ax1.set_title(f'{model_name}: Residual Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residual distribution
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residuals plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_importance(importance_dict, top_n=10, model_name='Model', save_path=None):
    """
    Plot feature importance.
    
    Args:
        importance_dict (dict): Dictionary of feature names to importance scores
        top_n (int): Number of top features to show
        model_name (str): Name of the model
        save_path (Path): Path to save the plot
    """
    logger.info(f"Creating feature importance plot for {model_name}")
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(features)), [abs(imp) for imp in importances])
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'{model_name}: Top {top_n} Most Important Features', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of multiple models.
    
    Args:
        results_dict (dict): Dictionary of model_name -> metrics
        save_path (Path): Path to save the plot
    """
    logger.info("Creating model comparison plot")
    
    models = list(results_dict.keys())
    rmse_values = [results_dict[m]['rmse'] for m in models]
    r2_values = [results_dict[m]['r2'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE comparison
    bars1 = ax1.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('RMSE ($)', fontsize=12)
    ax1.set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom')
    
    # R² comparison
    bars2 = ax2.bar(models, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_price_distribution(data, target_column='price', save_path=None):
    """
    Plot distribution of target variable (price).
    
    Args:
        data (pd.DataFrame): Dataset
        target_column (str): Name of target column
        save_path (Path): Path to save the plot
    """
    logger.info("Creating price distribution plot")
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data[target_column], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Price Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(data[target_column])
    plt.ylabel('Price ($)', fontsize=12)
    plt.title('Price Box Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved price distribution plot to {save_path}")
    
    plt.show()
    plt.close()
