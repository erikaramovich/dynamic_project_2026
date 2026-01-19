"""
Data loading utilities for the ML project.
Handles reading and basic validation of datasets.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(filepath, validate=True):
    """
    Load data from CSV file with optional validation.
    
    Args:
        filepath (str or Path): Path to the CSV file
        validate (bool): Whether to perform basic validation
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    
    if validate:
        validate_data(data)
    
    return data


def validate_data(data):
    """
    Perform basic validation on the dataset.
    
    Args:
        data (pd.DataFrame): Data to validate
        
    Raises:
        ValueError: If validation fails
    """
    if data.empty:
        raise ValueError("Dataset is empty")
    
    # Check for completely null columns
    null_columns = data.columns[data.isnull().all()].tolist()
    if null_columns:
        raise ValueError(f"Columns with all null values: {null_columns}")
    
    # Log warnings for columns with high null percentage
    null_percentages = data.isnull().sum() / len(data) * 100
    high_null_cols = null_percentages[null_percentages > 50].to_dict()
    
    if high_null_cols:
        logger.warning(f"Columns with >50% null values: {high_null_cols}")
    
    logger.info("Data validation passed")


def get_data_info(data):
    """
    Get detailed information about the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'null_counts': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    return info


def save_data(data, filepath):
    """
    Save DataFrame to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        filepath (str or Path): Path to save the CSV file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to {filepath}")
    data.to_csv(filepath, index=False)
    logger.info("Data saved successfully")
