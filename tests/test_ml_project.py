"""
Unit tests for the ML project.
Tests core functionality of data loading, preprocessing, and models.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_project.preprocessing import DataPreprocessor, split_data
from ml_project.data_loader import load_data, validate_data, get_data_info
from ml_project.evaluate import evaluate_model, calculate_percentage_error
from ml_project.models import LinearRegressionModel, RandomForestModel


@pytest.fixture
def sample_data():
    """Create sample housing data for testing."""
    data = pd.DataFrame({
        'square_feet': [1500, 2000, 2500, 3000, 1800],
        'bedrooms': [3, 4, 4, 5, 3],
        'bathrooms': [2, 2, 3, 3, 2],
        'year_built': [2010, 2015, 2018, 2020, 2012],
        'lot_size': [5000, 7000, 8000, 10000, 6000],
        'garage_spaces': [1, 2, 2, 3, 1],
        'neighborhood': ['Suburbs', 'Downtown', 'Suburbs', 'Waterfront', 'Rural'],
        'house_type': ['Single-Family', 'Condo', 'Single-Family', 'Single-Family', 'Ranch'],
        'price': [300000, 450000, 550000, 750000, 350000]
    })
    return data


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_validate_data_success(self, sample_data):
        """Test that valid data passes validation."""
        validate_data(sample_data)  # Should not raise
    
    def test_validate_data_empty(self):
        """Test that empty data fails validation."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Dataset is empty"):
            validate_data(empty_data)
    
    def test_get_data_info(self, sample_data):
        """Test data info extraction."""
        info = get_data_info(sample_data)
        assert info['shape'] == (5, 9)
        assert 'price' in info['columns']
        assert isinstance(info['memory_usage'], float)


class TestPreprocessing:
    """Tests for data preprocessing."""
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test fitting and transforming data."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert y.shape[0] == len(sample_data)
        assert preprocessor.feature_names is not None
    
    def test_split_data(self, sample_data):
        """Test train-test split."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
    
    def test_preprocessor_transform(self, sample_data):
        """Test transforming new data."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        # Transform without target
        new_data = sample_data.drop(columns=['price']).head(2)
        X_new = preprocessor.transform(new_data)
        
        assert X_new.shape[0] == 2
        assert X_new.shape[1] == X.shape[1]


class TestModels:
    """Tests for ML models."""
    
    def test_linear_regression_train_predict(self, sample_data):
        """Test Linear Regression training and prediction."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(predictions > 0)  # Prices should be positive
    
    def test_random_forest_train_predict(self, sample_data):
        """Test Random Forest training and prediction."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = RandomForestModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(predictions > 0)  # Prices should be positive
    
    def test_model_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        model = RandomForestModel()
        model.train(X, y)
        importance = model.get_feature_importance(preprocessor.feature_names)
        
        assert len(importance) == len(preprocessor.feature_names)
        assert all(v >= 0 for v in importance.values())


class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation metrics."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mae' in metrics
        assert len(predictions) == len(X_test)
    
    def test_percentage_error(self):
        """Test percentage error calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 320])
        
        errors = calculate_percentage_error(y_true, y_pred)
        
        assert len(errors) == 3
        assert all(errors >= 0)
        np.testing.assert_almost_equal(errors[0], 10.0, decimal=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
