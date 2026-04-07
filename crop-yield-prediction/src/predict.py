"""
Prediction and feature importance module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path
from typing import Dict, List, Union


def predict_crop_yield(model: Pipeline, input_dict: Dict[str, Union[float, str]],
                       feature_names: List[str]) -> float:
    """
    Predict crop yield for a single sample using input dictionary.
    
    Parameters
    ----------
    model : Pipeline
        Trained preprocessing + model pipeline
    input_dict : Dict
        Dictionary with feature values, e.g.:
        {
            "N": 90,
            "P": 40,
            "K": 40,
            "pH": 6.5,
            "organic_matter": 2.8,
            "rainfall": 220,
            "temp_min": 18,
            "temp_max": 31,
            "fertilizer_usage": 120,
            "crop_type": "rice"
        }
    feature_names : List[str]
        List of expected feature names (for validation)
        
    Returns
    -------
    float
        Predicted crop yield
        
    Raises
    ------
    ValueError
        If required fields are missing from input_dict
    """
    
    # Validate against the raw model input schema (pre-encoding feature names)
    required_fields = {
        'N', 'P', 'K', 'pH', 'organic_matter', 'rainfall',
        'temp_min', 'temp_max', 'fertilizer_usage', 'crop_type'
    }
    
    # Check if all required fields are present
    missing_fields = required_fields - set(input_dict.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Create DataFrame with single row
    df_input = pd.DataFrame([input_dict])
    
    # Make prediction
    prediction = model.predict(df_input)[0]
    
    return prediction


def get_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances from Random Forest model in pipeline.
    
    Parameters
    ----------
    model : Pipeline
        Trained pipeline
    feature_names : List[str]
        List of feature names after preprocessing
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importances sorted by importance
    """
    
    # Extract the Random Forest model from the pipeline
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['model']
    else:
        rf_model = model
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                           output_dir: str = "outputs") -> None:
    """
    Plot top N features by importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importance values
    top_n : int
        Number of top features to plot
    output_dir : str
        Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    
    # Color gradient for visual appeal
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances - Random Forest Model', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, v in enumerate(top_features['importance'].values):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_TOP{top_n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/feature_importance_TOP{top_n}.png")


def save_feature_importance_csv(importance_df: pd.DataFrame,
                               output_dir: str = "outputs") -> None:
    """
    Save feature importance to CSV file.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importance values
    output_dir : str
        Directory to save the CSV
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = f'{output_dir}/feature_importance.csv'
    importance_df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")


def print_top_features(importance_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print top N features and their importance scores.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importance values
    top_n : int
        Number of top features to print
    """
    print("\n" + "="*60)
    print(f"TOP {top_n} IMPORTANT FEATURES")
    print("="*60)
    
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{idx+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
    
    print("="*60 + "\n")


def predict_batch(model: Pipeline, df_batch: pd.DataFrame) -> np.ndarray:
    """
    Predict crop yield for multiple samples.
    
    Parameters
    ----------
    model : Pipeline
        Trained pipeline
    df_batch : pd.DataFrame
        DataFrame with features (without target column)
        
    Returns
    -------
    np.ndarray
        Array of predictions
    """
    predictions = model.predict(df_batch)
    return predictions


def create_prediction_results_df(df_features: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Create a results DataFrame with features and predictions.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Input features DataFrame
    predictions : np.ndarray
        Array of predictions
        
    Returns
    -------
    pd.DataFrame
        DataFrame combining features and predictions
    """
    results_df = df_features.copy()
    results_df['predicted_crop_yield'] = predictions
    return results_df
