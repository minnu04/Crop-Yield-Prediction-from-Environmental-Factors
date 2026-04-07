"""
Data preprocessing module.
Handles feature scaling, encoding, and imputation using sklearn transformers.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple


def build_preprocessor(df: pd.DataFrame, target: str = 'crop_yield',
                       categorical_col: str = 'crop_type') -> Tuple[ColumnTransformer, list]:
    """
    Build a ColumnTransformer for preprocessing with proper handling of all columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name to exclude from preprocessing
    categorical_col : str
        Categorical column name
        
    Returns
    -------
    Tuple[ColumnTransformer, list]
        Fitted preprocessor and feature names after transformation
    """
    
    # Get all columns except target
    all_columns = [col for col in df.columns if col != target]
    
    # Separate numerical and categorical columns
    numerical_cols = df[all_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[all_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define the preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define the preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keep any columns that are not explicitly handled
    )
    
    # Fit the preprocessor on the full dataset to get feature names
    preprocessor.fit(df[all_columns])
    
    # Get feature names after transformation
    feature_names = get_feature_names_after_transform(
        preprocessor, numerical_cols, categorical_cols
    )
    
    return preprocessor, feature_names


def get_feature_names_after_transform(preprocessor: ColumnTransformer, 
                                      numerical_cols: list,
                                      categorical_cols: list) -> list:
    """
    Get feature names after ColumnTransformer preprocessing.
    Handles one-hot encoded features correctly.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names
        
    Returns
    -------
    list
        Feature names after transformation
    """
    feature_names = []
    
    # Add numerical column names
    feature_names.extend(numerical_cols)
    
    # Add one-hot encoded categorical column names
    for col in categorical_cols:
        if hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
            onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categories = onehot.categories_[categorical_cols.index(col)]
            for cat in categories:
                feature_names.append(f'{col}_{cat}')
    
    return feature_names


def preprocess_data(preprocessor: ColumnTransformer, df: pd.DataFrame,
                   target: str = 'crop_yield') -> np.ndarray:
    """
    Apply preprocessing to data using a fitted preprocessor.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name to exclude
        
    Returns
    -------
    np.ndarray
        Preprocessed feature array
    """
    all_columns = [col for col in df.columns if col != target]
    return preprocessor.transform(df[all_columns])


def create_full_pipeline(model, preprocessor: ColumnTransformer,
                         target: str = 'crop_yield') -> Pipeline:
    """
    Create a full pipeline combining preprocessing and model.
    
    Parameters
    ----------
    model : estimator
        Sklearn model or pipeline
    preprocessor : ColumnTransformer
        Fitted preprocessor
    target : str
        Target column name
        
    Returns
    -------
    Pipeline
        Complete preprocessing + model pipeline
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
