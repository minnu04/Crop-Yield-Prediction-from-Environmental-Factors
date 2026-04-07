"""
Data loading and validation module.
Handles loading CSV data and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load crop yield data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV file is empty or cannot be read
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if df.empty:
        raise ValueError("Loaded dataset is empty")
    
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that all required columns are present in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    required_columns : list
        List of required column names
        
    Returns
    -------
    bool
        True if all required columns are present
        
    Raises
    ------
    ValueError
        If any required column is missing
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
        
    Returns
    -------
    dict
        Dictionary containing dataset info
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # in MB
    }
    return info
