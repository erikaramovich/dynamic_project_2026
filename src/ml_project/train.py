"""
Training script for house price prediction models.
Trains and evaluates all available models.
"""

import logging
import sys
from pathlib import Path

try:
    from .config import (
        HOUSING_DATA_PATH,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE,
        VALIDATION_SIZE
    )
    from .data_loader import load_data
    from .preprocessing import DataPreprocessor, split_data
    from .models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel
    from .evaluate import evaluate_model, compare_models, print_prediction_summary
except ImportError:
    from config import (
        HOUSING_DATA_PATH,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE,
        VALIDATION_SIZE
    )
    from data_loader import load_data
    from preprocessing import DataPreprocessor, split_data
    from models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel
    from evaluate import evaluate_model, compare_models, print_prediction_summary

# Set up logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main training pipeline.
    
    Steps:
    1. Load data
    2. Preprocess and split data
    3. Train multiple models
    4. Evaluate and compare models
    5. Save the best models
    """
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("="*70 + "\n")
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        if not HOUSING_DATA_PATH.exists():
            print(f"❌ Error: Data file not found at {HOUSING_DATA_PATH}")
            print("Please run: python src/ml_project/generate_data.py")
            return
        
        data = load_data(HOUSING_DATA_PATH)
        print(f"✅ Loaded {len(data)} samples")
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(data)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = split_data(
            X_train, y_train, 
            test_size=VALIDATION_SIZE
        )
        
        print(f"✅ Preprocessed data: {X.shape[1]} features")
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Save preprocessor
        preprocessor.save()
        print(f"✅ Saved preprocessor")
        
        # Step 3: Train models
        print("\n" + "-"*70)
        print("TRAINING MODELS")
        print("-"*70 + "\n")
        
        results = {}
        trained_models = {}
        
        # Train Linear Regression
        print("1. Training Linear Regression...")
        lr_model = LinearRegressionModel()
        lr_model.train(X_train, y_train)
        metrics, y_pred = evaluate_model(lr_model, X_test, y_test)
        results['Linear Regression'] = metrics
        trained_models['Linear Regression'] = lr_model
        lr_model.save()
        print("✅ Linear Regression trained and saved")
        
        # Train Random Forest
        print("\n2. Training Random Forest...")
        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train)
        metrics, y_pred = evaluate_model(rf_model, X_test, y_test)
        results['Random Forest'] = metrics
        trained_models['Random Forest'] = rf_model
        rf_model.save()
        print("✅ Random Forest trained and saved")
        
        # Train Neural Network
        print("\n3. Training Neural Network...")
        nn_model = NeuralNetworkModel()
        nn_model.build_model(X_train.shape[1])
        nn_model.train(X_train, y_train, X_val, y_val, verbose=0)
        metrics, y_pred = evaluate_model(nn_model, X_test, y_test)
        results['Neural Network'] = metrics
        trained_models['Neural Network'] = nn_model
        nn_model.save()
        print("✅ Neural Network trained and saved")
        
        # Step 4: Compare models
        print("\n" + "-"*70)
        best_model_name = compare_models(results)
        
        # Print detailed summary for best model
        best_model = trained_models[best_model_name]
        _, best_predictions = evaluate_model(best_model, X_test, y_test)
        print_prediction_summary(y_test, best_predictions, best_model_name)
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'get_feature_importance'):
            print(f"\nTop 5 Most Important Features ({best_model_name}):")
            print("-" * 50)
            importance = best_model.get_feature_importance(preprocessor.feature_names)
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, score) in enumerate(sorted_features[:5], 1):
                print(f"{i}. {feature:<20} {abs(score):>12.4f}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nAll models saved to: models/")
        print("\nNext steps:")
        print("  - Make predictions: python src/ml_project/predict.py")
        print("  - Explore in notebook: jupyter notebook notebooks/ml_tutorial.ipynb")
        print("  - Visualize results: python src/ml_project/visualize.py")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error during training: {str(e)}")
        print("Check logs/ml_project.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
