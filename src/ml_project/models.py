"""
Model implementations for house price prediction.
Demonstrates three different ML approaches: Linear Regression, Random Forest, and Neural Networks.
"""

import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

try:
    from .config import (
        RANDOM_SEED,
        RF_N_ESTIMATORS,
        RF_MAX_DEPTH,
        RF_MIN_SAMPLES_SPLIT,
        NN_EPOCHS,
        NN_BATCH_SIZE,
        NN_LEARNING_RATE,
        NN_HIDDEN_LAYERS,
        LINEAR_MODEL_PATH,
        RF_MODEL_PATH,
        NN_MODEL_PATH
    )
except ImportError:
    from config import (
        RANDOM_SEED,
        RF_N_ESTIMATORS,
        RF_MAX_DEPTH,
        RF_MIN_SAMPLES_SPLIT,
        NN_EPOCHS,
        NN_BATCH_SIZE,
        NN_LEARNING_RATE,
        NN_HIDDEN_LAYERS,
        LINEAR_MODEL_PATH,
        RF_MODEL_PATH,
        NN_MODEL_PATH
    )

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class LinearRegressionModel:
    """
    Simple Linear Regression model.
    Best for understanding baseline performance and feature relationships.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.name = "Linear Regression"
        
    def train(self, X_train, y_train):
        """Train the linear regression model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.name} training completed")
        
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature coefficients (importance)."""
        return dict(zip(feature_names, self.model.coef_))
    
    def save(self, filepath=LINEAR_MODEL_PATH):
        """Save model to disk."""
        logger.info(f"Saving {self.name} to {filepath}")
        joblib.dump(self.model, filepath)
        
    @classmethod
    def load(cls, filepath=LINEAR_MODEL_PATH):
        """Load model from disk."""
        logger.info(f"Loading Linear Regression from {filepath}")
        model_obj = cls()
        model_obj.model = joblib.load(filepath)
        return model_obj


class RandomForestModel:
    """
    Random Forest Regressor.
    Ensemble method that often performs well without much tuning.
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RANDOM_SEED,
            n_jobs=-1  # Use all CPU cores
        )
        self.name = "Random Forest"
        
    def train(self, X_train, y_train):
        """Train the random forest model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.name} training completed")
        
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores."""
        return dict(zip(feature_names, self.model.feature_importances_))
    
    def save(self, filepath=RF_MODEL_PATH):
        """Save model to disk."""
        logger.info(f"Saving {self.name} to {filepath}")
        joblib.dump(self.model, filepath)
        
    @classmethod
    def load(cls, filepath=RF_MODEL_PATH):
        """Load model from disk."""
        logger.info(f"Loading Random Forest from {filepath}")
        model_obj = cls()
        model_obj.model = joblib.load(filepath)
        return model_obj


class NeuralNetworkModel:
    """
    Neural Network using TensorFlow/Keras.
    Demonstrates deep learning for regression tasks.
    """
    
    def __init__(self, input_dim=None):
        self.model = None
        self.name = "Neural Network"
        self.input_dim = input_dim
        
    def build_model(self, input_dim):
        """
        Build the neural network architecture.
        
        Architecture:
        - Input layer
        - Multiple hidden layers with ReLU activation
        - Dropout for regularization
        - Output layer (single neuron for regression)
        """
        self.input_dim = input_dim
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
        ])
        
        # Add hidden layers
        for i, units in enumerate(NN_HIDDEN_LAYERS):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
            model.add(layers.Dropout(0.2, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built {self.name} with {len(NN_HIDDEN_LAYERS)} hidden layers")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        logger.info(f"Training {self.name}...")
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        logger.info(f"{self.name} training completed")
        return history
        
    def predict(self, X):
        """Make predictions."""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def save(self, filepath=NN_MODEL_PATH):
        """Save model to disk."""
        logger.info(f"Saving {self.name} to {filepath}")
        self.model.save(filepath)
        
    @classmethod
    def load(cls, filepath=NN_MODEL_PATH):
        """Load model from disk."""
        logger.info(f"Loading Neural Network from {filepath}")
        model_obj = cls()
        model_obj.model = keras.models.load_model(filepath)
        return model_obj


def get_all_models(input_dim=None):
    """
    Get instances of all available models.
    
    Args:
        input_dim (int): Number of input features (required for neural network)
        
    Returns:
        dict: Dictionary of model name to model instance
    """
    models = {
        'linear_regression': LinearRegressionModel(),
        'random_forest': RandomForestModel(),
    }
    
    if input_dim is not None:
        models['neural_network'] = NeuralNetworkModel(input_dim)
    
    return models
