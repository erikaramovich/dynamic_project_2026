"""
Data generation script to create synthetic housing data for learning purposes.
This creates realistic house price data with various features.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

try:
    from .config import (
        HOUSING_DATA_PATH,
        RANDOM_SEED,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE
    )
except ImportError:
    from config import (
        HOUSING_DATA_PATH,
        RANDOM_SEED,
        LOG_LEVEL,
        LOG_FORMAT,
        LOG_FILE
    )

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


def generate_housing_data(n_samples=1000, seed=RANDOM_SEED):
    """
    Generate synthetic housing data for machine learning practice.
    
    This function creates realistic house price data with correlations between features.
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic housing data
    """
    logger.info(f"Generating {n_samples} housing samples with seed {seed}")
    
    np.random.seed(seed)
    
    # Generate base features
    square_feet = np.random.normal(2000, 500, n_samples).clip(800, 5000)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    year_built = np.random.randint(1950, 2024, n_samples)
    lot_size = np.random.normal(8000, 3000, n_samples).clip(3000, 20000)
    garage_spaces = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1])
    
    # Categorical features
    neighborhoods = ['Downtown', 'Suburbs', 'Rural', 'Waterfront', 'Historic']
    neighborhood = np.random.choice(neighborhoods, n_samples)
    
    house_types = ['Single-Family', 'Townhouse', 'Condo', 'Ranch']
    house_type = np.random.choice(house_types, n_samples)
    
    # Generate price with realistic correlations
    # Base price influenced by multiple factors
    base_price = 100000
    
    # Square footage is the primary driver (strong correlation)
    price = base_price + square_feet * 150
    
    # Add other feature influences
    price += bedrooms * 20000
    price += bathrooms * 15000
    price += (2024 - year_built) * (-500)  # Newer homes are more expensive
    price += lot_size * 5
    price += garage_spaces * 10000
    
    # Neighborhood premium
    neighborhood_premium = {
        'Downtown': 50000,
        'Suburbs': 20000,
        'Rural': -10000,
        'Waterfront': 100000,
        'Historic': 30000
    }
    price += np.array([neighborhood_premium[n] for n in neighborhood])
    
    # House type adjustment
    house_type_adjustment = {
        'Single-Family': 20000,
        'Townhouse': 0,
        'Condo': -15000,
        'Ranch': 10000
    }
    price += np.array([house_type_adjustment[h] for h in house_type])
    
    # Add some random noise to make it realistic
    price += np.random.normal(0, 30000, n_samples)
    
    # Ensure no negative prices
    price = price.clip(50000, None)
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_feet': square_feet.astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'lot_size': lot_size.astype(int),
        'garage_spaces': garage_spaces,
        'neighborhood': neighborhood,
        'house_type': house_type,
        'price': price.astype(int)
    })
    
    logger.info(f"Generated data shape: {data.shape}")
    logger.info(f"Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
    logger.info(f"Average price: ${data['price'].mean():,.0f}")
    
    return data


def save_data(data, filepath):
    """
    Save DataFrame to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        filepath (Path): Path to save the CSV file
    """
    logger.info(f"Saving data to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    logger.info(f"Data saved successfully")


def main():
    """
    Main function to generate and save housing data.
    """
    logger.info("Starting data generation...")
    
    # Generate data
    data = generate_housing_data(n_samples=1000)
    
    # Display basic statistics
    print("\n" + "="*60)
    print("HOUSING DATA GENERATED")
    print("="*60)
    print(f"\nDataset shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nBasic statistics:")
    print(data.describe())
    print(f"\nData types:")
    print(data.dtypes)
    print("\n" + "="*60)
    
    # Save to file
    save_data(data, HOUSING_DATA_PATH)
    
    print(f"\n✅ Data saved to: {HOUSING_DATA_PATH}")
    print("You can now run: python src/ml_project/train.py")
    
    logger.info("Data generation completed successfully")


if __name__ == "__main__":
    main()
