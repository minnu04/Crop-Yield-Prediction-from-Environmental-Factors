"""
Model evaluation module.
Computes and displays evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing RMSE, MAE, and R² score
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mape = compute_mape(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    return metrics


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE value
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Evaluate model on both train and test sets.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
        
    Returns
    -------
    Dict
        Dictionary containing train and test metrics
    """
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    
    results = {
        'train': train_metrics,
        'test': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return results


def print_evaluation_report(eval_results: Dict) -> None:
    """
    Print comprehensive evaluation report.
    
    Parameters
    ----------
    eval_results : Dict
        Dictionary containing train and test metrics
    """
    train_metrics = eval_results['train']
    test_metrics = eval_results['test']
    
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)
    
    # Create comparison table
    print("\n{:<20} {:<20} {:<20}".format("Metric", "Train Set", "Test Set"))
    print("-"*70)
    
    print("{:<20} {:<20.4f} {:<20.4f}".format(
        "R² Score",
        train_metrics['r2'],
        test_metrics['r2']
    ))
    
    print("{:<20} {:<20.4f} {:<20.4f}".format(
        "RMSE",
        train_metrics['rmse'],
        test_metrics['rmse']
    ))
    
    print("{:<20} {:<20.4f} {:<20.4f}".format(
        "MAE",
        train_metrics['mae'],
        test_metrics['mae']
    ))
    
    print("{:<20} {:<20.2f}% {:<20.2f}%".format(
        "MAPE",
        train_metrics['mape'],
        test_metrics['mape']
    ))
    
    print("-"*70)
    
    # Overfitting analysis
    print("\nOVERFITTING ANALYSIS:")
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    rmse_diff_pct = ((test_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse']) * 100
    
    print(f"  R² Difference (Train - Test): {r2_diff:.4f}", end="")
    if abs(r2_diff) < 0.05:
        print(" ✓ (Good - minimal overfitting)")
    elif abs(r2_diff) < 0.15:
        print(" ⚠ (Moderate overfitting)")
    else:
        print(" ✗ (Significant overfitting)")
    
    print(f"  RMSE Difference: {rmse_diff_pct:.2f}%", end="")
    if rmse_diff_pct < 10:
        print(" ✓ (Good)")
    elif rmse_diff_pct < 25:
        print(" ⚠ (Moderate)")
    else:
        print(" ✗ (Significant)")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'r2_train': train_metrics['r2'],
        'r2_test': test_metrics['r2'],
        'rmse_train': train_metrics['rmse'],
        'rmse_test': test_metrics['rmse'],
        'mae_train': train_metrics['mae'],
        'mae_test': test_metrics['mae']
    }


def check_for_overfitting(eval_results: Dict) -> bool:
    """
    Check if model shows significant overfitting.
    
    Parameters
    ----------
    eval_results : Dict
        Dictionary containing train and test metrics
        
    Returns
    -------
    bool
        True if overfitting is detected
    """
    r2_diff = eval_results['train']['r2'] - eval_results['test']['r2']
    rmse_diff = eval_results['test']['rmse'] - eval_results['train']['rmse']
    
    # Consider overfitting if R² drops by more than 0.15 or RMSE increases significantly
    return r2_diff > 0.15 or rmse_diff > (eval_results['train']['rmse'] * 0.25)
