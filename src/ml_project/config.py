"""
Configuration file for ML project.
Centralizes all hyperparameters and settings for easy experimentation.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
HOUSING_DATA_PATH = RAW_DATA_DIR / "housing_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_housing_data.csv"

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Linear Regression settings
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression.pkl"

# Random Forest settings
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"

# Neural Network settings
NN_EPOCHS = 100
NN_BATCH_SIZE = 32
NN_LEARNING_RATE = 0.001
NN_HIDDEN_LAYERS = [64, 32, 16]
NN_MODEL_PATH = MODELS_DIR / "neural_network.keras"

# Preprocessing settings
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Features for house price prediction
NUMERICAL_FEATURES = [
    'square_feet',
    'bedrooms',
    'bathrooms',
    'year_built',
    'lot_size',
    'garage_spaces'
]

CATEGORICAL_FEATURES = [
    'neighborhood',
    'house_type'
]

TARGET_COLUMN = 'price'

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "ml_project.log"
