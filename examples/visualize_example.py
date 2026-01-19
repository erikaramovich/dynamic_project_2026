"""
Example visualization script demonstrating data exploration.
Run this after training models to see visualizations of results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import matplotlib.pyplot as plt
import seaborn as sns
from ml_project.data_loader import load_data
from ml_project.config import HOUSING_DATA_PATH, RF_MODEL_PATH, SCALER_PATH
from ml_project.preprocessing import DataPreprocessor, split_data
from ml_project.models import RandomForestModel
from ml_project.visualize import (
    plot_feature_correlations,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_price_distribution
)
from ml_project.evaluate import evaluate_model


def main():
    """
    Create visualizations for the ML project.
    """
    print("\n" + "="*70)
    print("ML PROJECT - DATA VISUALIZATION")
    print("="*70 + "\n")
    
    # Check if data exists
    if not HOUSING_DATA_PATH.exists():
        print("❌ Data not found. Please run: python src/ml_project/generate_data.py")
        return
    
    # Load data
    print("Loading data...")
    data = load_data(HOUSING_DATA_PATH)
    print(f"✅ Loaded {len(data)} samples\n")
    
    # 1. Price Distribution
    print("1. Creating price distribution plot...")
    plot_price_distribution(data)
    
    # 2. Feature Correlations
    print("2. Creating correlation heatmap...")
    plot_feature_correlations(data)
    
    # Check if model exists
    if not RF_MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("\n❌ Trained models not found.")
        print("Please run: python src/ml_project/train.py")
        print("\nShowing only data visualizations.")
        return
    
    # Load preprocessor and split data
    print("\n3. Loading model and making predictions...")
    preprocessor = DataPreprocessor.load(SCALER_PATH)
    X, y = preprocessor.fit_transform(data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Load model
    model = RandomForestModel.load(RF_MODEL_PATH)
    
    # Evaluate
    metrics, predictions = evaluate_model(model, X_test, y_test)
    print(f"   Model RMSE: ${metrics['rmse']:,.2f}")
    print(f"   Model R²: {metrics['r2']:.4f}\n")
    
    # 4. Predictions vs Actual
    print("4. Creating predictions vs actual plot...")
    plot_predictions_vs_actual(y_test, predictions, 'Random Forest')
    
    # 5. Residuals
    print("5. Creating residual plots...")
    plot_residuals(y_test, predictions, 'Random Forest')
    
    # 6. Feature Importance
    print("6. Creating feature importance plot...")
    importance = model.get_feature_importance(preprocessor.feature_names)
    plot_feature_importance(importance, top_n=8, model_name='Random Forest')
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETED!")
    print("="*70)
    print("\nAll plots displayed. Close the plot windows to continue.")
    print("\nTips:")
    print("  - Examine correlations to understand feature relationships")
    print("  - Check residuals for patterns (should be random)")
    print("  - Use feature importance to guide feature engineering")


if __name__ == "__main__":
    main()
