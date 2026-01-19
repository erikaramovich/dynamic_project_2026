"""
Prediction script for making predictions with trained models.
Demonstrates how to load and use trained models for inference.
"""

import logging
import sys
import numpy as np
import pandas as pd

try:
    from .config import (
        LINEAR_MODEL_PATH,
        RF_MODEL_PATH,
        NN_MODEL_PATH,
        SCALER_PATH,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE
    )
    from .models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel
    from .preprocessing import DataPreprocessor
except ImportError:
    from config import (
        LINEAR_MODEL_PATH,
        RF_MODEL_PATH,
        NN_MODEL_PATH,
        SCALER_PATH,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE
    )
    from models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel
    from preprocessing import DataPreprocessor

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


def predict_single_house(house_data, model_type='random_forest'):
    """
    Predict the price of a single house.
    
    Args:
        house_data (dict): Dictionary containing house features
        model_type (str): Type of model to use ('linear_regression', 'random_forest', 'neural_network')
        
    Returns:
        float: Predicted price
    """
    logger.info(f"Making prediction with {model_type}")
    
    # Load preprocessor
    preprocessor = DataPreprocessor.load(SCALER_PATH)
    
    # Convert input to DataFrame
    df = pd.DataFrame([house_data])
    
    # Preprocess
    X = preprocessor.transform(df)
    
    # Load appropriate model and predict
    if model_type == 'linear_regression':
        model = LinearRegressionModel.load(LINEAR_MODEL_PATH)
    elif model_type == 'random_forest':
        model = RandomForestModel.load(RF_MODEL_PATH)
    elif model_type == 'neural_network':
        model = NeuralNetworkModel.load(NN_MODEL_PATH)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    prediction = model.predict(X)[0]
    
    return prediction


def predict_batch(data_df, model_type='random_forest'):
    """
    Predict prices for multiple houses.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing house features
        model_type (str): Type of model to use
        
    Returns:
        np.ndarray: Predicted prices
    """
    logger.info(f"Making batch predictions with {model_type}")
    
    # Load preprocessor
    preprocessor = DataPreprocessor.load(SCALER_PATH)
    
    # Preprocess
    X = preprocessor.transform(data_df)
    
    # Load appropriate model and predict
    if model_type == 'linear_regression':
        model = LinearRegressionModel.load(LINEAR_MODEL_PATH)
    elif model_type == 'random_forest':
        model = RandomForestModel.load(RF_MODEL_PATH)
    elif model_type == 'neural_network':
        model = NeuralNetworkModel.load(NN_MODEL_PATH)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    predictions = model.predict(X)
    
    return predictions


def demo_predictions():
    """
    Demonstrate predictions with example houses.
    """
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - DEMO")
    print("="*70 + "\n")
    
    # Example houses with different characteristics
    example_houses = [
        {
            'square_feet': 2500,
            'bedrooms': 4,
            'bathrooms': 2.5,
            'year_built': 2015,
            'lot_size': 10000,
            'garage_spaces': 2,
            'neighborhood': 'Suburbs',
            'house_type': 'Single-Family'
        },
        {
            'square_feet': 1800,
            'bedrooms': 3,
            'bathrooms': 2,
            'year_built': 2010,
            'lot_size': 6000,
            'garage_spaces': 1,
            'neighborhood': 'Downtown',
            'house_type': 'Condo'
        },
        {
            'square_feet': 3500,
            'bedrooms': 5,
            'bathrooms': 3,
            'year_built': 2020,
            'lot_size': 15000,
            'garage_spaces': 3,
            'neighborhood': 'Waterfront',
            'house_type': 'Single-Family'
        }
    ]
    
    model_types = ['linear_regression', 'random_forest', 'neural_network']
    model_names = ['Linear Regression', 'Random Forest', 'Neural Network']
    
    try:
        for i, house in enumerate(example_houses, 1):
            print(f"House {i}:")
            print("-" * 70)
            print(f"  Square Feet: {house['square_feet']:,}")
            print(f"  Bedrooms: {house['bedrooms']}")
            print(f"  Bathrooms: {house['bathrooms']}")
            print(f"  Year Built: {house['year_built']}")
            print(f"  Lot Size: {house['lot_size']:,} sq ft")
            print(f"  Garage Spaces: {house['garage_spaces']}")
            print(f"  Neighborhood: {house['neighborhood']}")
            print(f"  House Type: {house['house_type']}")
            print("\nPredicted Prices:")
            
            for model_type, model_name in zip(model_types, model_names):
                try:
                    price = predict_single_house(house, model_type)
                    print(f"  {model_name:<20} ${price:>12,.2f}")
                except Exception as e:
                    print(f"  {model_name:<20} Error: {str(e)}")
            
            print()
        
        print("="*70)
        print("DEMO COMPLETED")
        print("="*70)
        print("\nTo make your own predictions, use the predict_single_house() function")
        print("or modify this script with your own house data.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required model file not found")
        print("Please run training first: python src/ml_project/train.py")
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main function to run prediction demo.
    """
    demo_predictions()


if __name__ == "__main__":
    main()
