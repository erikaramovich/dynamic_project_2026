"""
Data preprocessing module for feature engineering and transformation.
Demonstrates important ML preprocessing concepts.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

try:
    from .config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_SEED,
        SCALER_PATH
    )
except ImportError:
    from config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_SEED,
        SCALER_PATH
    )

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing operations including:
    - Feature encoding
    - Feature scaling
    - Train-test splitting
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def fit_transform(self, data):
        """
        Fit preprocessor on training data and transform it.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            tuple: (X_scaled, y) - Preprocessed features and target
        """
        logger.info("Fitting and transforming data")
        
        # Separate features and target
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN].values
        
        # Handle categorical features
        X_encoded = self._encode_categorical(X, fit=True)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        self.feature_names = X_encoded.columns.tolist()
        logger.info(f"Fitted preprocessor with {len(self.feature_names)} features")
        
        return X_scaled, y
    
    def transform(self, data):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data (pd.DataFrame): Raw data (without target)
            
        Returns:
            np.ndarray: Preprocessed features
        """
        logger.info("Transforming data")
        
        # Handle categorical features
        X_encoded = self._encode_categorical(data, fit=False)
        
        # Scale numerical features
        X_scaled = self.scaler.transform(X_encoded)
        
        return X_scaled
    
    def _encode_categorical(self, X, fit=False):
        """
        Encode categorical features using Label Encoding.
        
        Note for learners:
        - Label Encoding: Assigns integers to categories (e.g., 'A'->0, 'B'->1)
          - Simple and memory-efficient
          - Works well with tree-based models (Random Forest, XGBoost)
          - May not work well with linear models (implies ordinal relationship)
        
        - One-Hot Encoding: Creates binary columns for each category
          - Better for linear models
          - More memory-intensive
          - Can be added by setting use_onehot=True in config
        
        For production systems, consider:
        - One-Hot Encoding for linear models
        - Target Encoding for high-cardinality features
        - Feature hashing for very large categorical spaces
        
        Args:
            X (pd.DataFrame): Features
            fit (bool): Whether to fit the encoders
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_copy = X.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col in X_copy.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X_copy[col] = self.label_encoders[col].fit_transform(X_copy[col])
                else:
                    X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        
        return X_copy
    
    def save(self, filepath=SCALER_PATH):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath (Path): Path to save the preprocessor
        """
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
        logger.info("Preprocessor saved successfully")
    
    @classmethod
    def load(cls, filepath=SCALER_PATH):
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath (Path): Path to load the preprocessor from
            
        Returns:
            DataPreprocessor: Loaded preprocessor
        """
        logger.info(f"Loading preprocessor from {filepath}")
        saved_data = joblib.load(filepath)
        
        preprocessor = cls()
        preprocessor.scaler = saved_data['scaler']
        preprocessor.label_encoders = saved_data['label_encoders']
        preprocessor.feature_names = saved_data['feature_names']
        
        logger.info("Preprocessor loaded successfully")
        return preprocessor


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray or pd.DataFrame): Features
        y (np.ndarray): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def create_feature_summary(data):
    """
    Create a summary of features in the dataset.
    
    Args:
        data (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Feature summary
    """
    summary = pd.DataFrame({
        'dtype': data.dtypes,
        'null_count': data.isnull().sum(),
        'null_percentage': (data.isnull().sum() / len(data) * 100).round(2),
        'unique_values': data.nunique(),
        'sample_value': data.iloc[0]
    })
    
    return summary
