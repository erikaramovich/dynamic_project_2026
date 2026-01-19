"""
Model evaluation utilities.
Provides functions to assess model performance using various metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name=None):
    """
    Evaluate a model on test data using multiple metrics.
    
    Metrics explained:
    - MSE (Mean Squared Error): Average squared difference between predictions and actual values
    - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as target
    - MAE (Mean Absolute Error): Average absolute difference
    - R² Score: Proportion of variance explained by the model (1.0 is perfect)
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: True test values
        model_name (str): Name of the model for logging
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    name = model_name or getattr(model, 'name', 'Model')
    logger.info(f"Evaluating {name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"{name} - RMSE: {rmse:,.2f}, R²: {r2:.4f}")
    
    return metrics, y_pred


def compare_models(results):
    """
    Compare multiple models and determine the best one.
    
    Args:
        results (dict): Dictionary of model_name -> metrics
        
    Returns:
        str: Name of the best model (lowest RMSE)
    """
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R² Score':>12}")
    print("-"*70)
    
    best_model = None
    best_rmse = float('inf')
    
    for model_name, metrics in results.items():
        rmse = metrics['rmse']
        mae = metrics['mae']
        r2 = metrics['r2']
        
        print(f"{model_name:<20} ${rmse:>11,.2f} ${mae:>11,.2f} {r2:>12.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_name
    
    print("-"*70)
    print(f"Best Model: {best_model} (RMSE: ${best_rmse:,.2f})")
    print("="*70 + "\n")
    
    return best_model


def calculate_percentage_error(y_true, y_pred):
    """
    Calculate percentage errors for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        np.ndarray: Percentage errors
    """
    return np.abs((y_true - y_pred) / y_true) * 100


def get_prediction_summary(y_true, y_pred):
    """
    Get a summary of prediction accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Summary statistics
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = calculate_percentage_error(y_true, y_pred)
    
    summary = {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors),
        'mean_pct_error': np.mean(pct_errors),
        'median_pct_error': np.median(pct_errors),
        'within_10_pct': np.sum(pct_errors <= 10) / len(pct_errors) * 100,
        'within_20_pct': np.sum(pct_errors <= 20) / len(pct_errors) * 100,
    }
    
    return summary


def print_prediction_summary(y_true, y_pred, model_name="Model"):
    """
    Print a formatted prediction summary.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name of the model
    """
    summary = get_prediction_summary(y_true, y_pred)
    
    print(f"\n{model_name} Prediction Summary:")
    print("-" * 50)
    print(f"Mean Error:              ${summary['mean_error']:>12,.2f}")
    print(f"Mean Absolute Error:     ${summary['mean_abs_error']:>12,.2f}")
    print(f"Median Absolute Error:   ${summary['median_abs_error']:>12,.2f}")
    print(f"Mean % Error:            {summary['mean_pct_error']:>12.2f}%")
    print(f"Median % Error:          {summary['median_pct_error']:>12.2f}%")
    print(f"Predictions within 10%:  {summary['within_10_pct']:>12.1f}%")
    print(f"Predictions within 20%:  {summary['within_20_pct']:>12.1f}%")
    print("-" * 50)
