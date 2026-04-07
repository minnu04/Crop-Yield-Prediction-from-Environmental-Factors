"""
Model training and hyperparameter tuning module.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, Dict, Any


def split_data(df: pd.DataFrame, target: str = 'crop_yield', 
               test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split dataset into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : str
        Target column name
    test_size : float
        Proportion of dataset to include in test set
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series,
                        preprocessor: ColumnTransformer,
                        random_state: int = 42) -> Pipeline:
    """
    Train a baseline Random Forest model without tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    preprocessor : ColumnTransformer
        Fitted preprocessor
    random_state : int
        Random seed
        
    Returns
    -------
    Pipeline
        Trained baseline model pipeline
    """
    # Create base model
    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    print("\n" + "="*60)
    print("TRAINING BASELINE MODEL")
    print("="*60)
    print("Model: Random Forest Regressor")
    print(f"Training samples: {len(X_train)}")
    print("Training in progress...")
    
    pipeline.fit(X_train, y_train)
    
    print("✓ Baseline model training completed!")
    print("="*60 + "\n")
    
    return pipeline


def tune_model(X_train: pd.DataFrame, y_train: pd.Series,
              preprocessor: ColumnTransformer,
              cv_folds: int = 5,
              random_state: int = 42) -> Tuple[Pipeline, Dict]:
    """
    Tune Random Forest hyperparameters using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    preprocessor : ColumnTransformer
        Fitted preprocessor
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[Pipeline, Dict]
        Best trained pipeline and grid search results
    """
    
    # Define hyperparameter grid
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }
    
    # Base model
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*60)
    print(f"Parameter Grid: {param_grid}")
    print(f"Cross-validation folds: {cv_folds}")
    print("Tuning in progress...")
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n✓ Tuning completed!")
    print(f"Best CV R² Score: {best_score:.4f}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    print("="*60 + "\n")
    
    # Return best model and results
    results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'cv_results': grid_search.cv_results_,
        'grid_search': grid_search
    }
    
    return grid_search.best_estimator_, results


def perform_cross_validation(pipeline: Pipeline, X_train: pd.DataFrame,
                             y_train: pd.Series, cv_folds: int = 5) -> Dict:
    """
    Perform cross-validation to assess model stability.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    Dict
        Cross-validation scores
    """
    cv_results = {}
    
    # R² scores
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2')
    cv_results['r2_scores'] = r2_scores
    cv_results['r2_mean'] = r2_scores.mean()
    cv_results['r2_std'] = r2_scores.std()
    
    # RMSE scores (negative because sklearn uses neg_mean_squared_error)
    rmse_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv_folds, 
                                    scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(rmse_scores)
    cv_results['rmse_scores'] = rmse_scores
    cv_results['rmse_mean'] = rmse_scores.mean()
    cv_results['rmse_std'] = rmse_scores.std()
    
    # MAE scores
    mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv_folds,
                                   scoring='neg_mean_absolute_error')
    cv_results['mae_scores'] = mae_scores
    cv_results['mae_mean'] = mae_scores.mean()
    cv_results['mae_std'] = mae_scores.std()
    
    return cv_results
