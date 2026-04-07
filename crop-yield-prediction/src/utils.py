"""
Utility functions for the crop yield prediction project.
"""

import os
import json
from pathlib import Path
import joblib


def create_directories(paths: list) -> None:
    """
    Create directories if they don't exist.
    
    Parameters
    ----------
    paths : list
        List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_dict_to_json(data: dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Parameters
    ----------
    data : dict
        Dictionary to save
    filepath : str
        Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_dict_from_json(filepath: str) -> dict:
    """
    Load a dictionary from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file
        
    Returns
    -------
    dict
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_model(model, filepath: str) -> None:
    """
    Save a trained model/pipeline to disk using joblib.

    Parameters
    ----------
    model : Any
        Trained model or pipeline object
    filepath : str
        Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Load a trained model/pipeline from disk.

    Parameters
    ----------
    filepath : str
        Path to model file

    Returns
    -------
    Any
        Loaded model object
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def validate_input_dict(input_dict: dict, required_fields: list) -> bool:
    """
    Validate that all required fields are present in input dictionary.
    
    Parameters
    ----------
    input_dict : dict
        Input dictionary to validate
    required_fields : list
        List of required field names
        
    Returns
    -------
    bool
        True if all required fields are present, False otherwise
    """
    return all(field in input_dict for field in required_fields)


def get_relative_path(base_path: str, target_path: str) -> str:
    """
    Get relative path from base to target.
    
    Parameters
    ----------
    base_path : str
        Base path
    target_path : str
        Target path
        
    Returns
    -------
    str
        Relative path
    """
    return os.path.relpath(target_path, base_path)
