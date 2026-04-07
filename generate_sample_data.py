"""
Generate synthetic crop yield data for testing and development.
Creates a CSV file with realistic agricultural data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples: int = 1000, output_path: str = 'data/crop_yield_data.csv',
                        random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic crop yield dataset.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    output_path : str
        Path to save the CSV file
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)
    
    # Define crop types
    crop_types = ['rice', 'maize', 'wheat', 'barley', 'oats']
    
    # Generate features with realistic ranges
    data = {
        'N': np.random.uniform(20, 100, n_samples),  # Nitrogen (kg/ha)
        'P': np.random.uniform(5, 80, n_samples),    # Phosphorus (kg/ha)
        'K': np.random.uniform(5, 80, n_samples),    # Potassium (kg/ha)
        'pH': np.random.uniform(5.5, 8.5, n_samples),  # pH level
        'organic_matter': np.random.uniform(1.0, 5.0, n_samples),  # OM (%)
        'rainfall': np.random.uniform(100, 300, n_samples),  # mm
        'temp_min': np.random.uniform(10, 25, n_samples),  # °C
        'temp_max': np.random.uniform(25, 40, n_samples),  # °C
        'fertilizer_usage': np.random.uniform(50, 200, n_samples),  # kg/ha
        'crop_type': np.random.choice(crop_types, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic crop yield based on features (with some noise)
    # Higher N, P, K, and organic matter generally lead to higher yield
    # Optimal pH and temperature ranges exist
    # Moderate rainfall is beneficial
    
    base_yield = (
        df['N'] * 0.2 +
        df['P'] * 0.3 +
        df['K'] * 0.25 +
        (6.5 - np.abs(df['pH'] - 6.5)) * 5 +  # Penalty for deviance from optimal pH
        df['organic_matter'] * 8 +
        df['rainfall'] * 0.15 +
        np.clip(df['temp_min'], 15, 25) * 0.5 +  # Optimal temp range
        np.clip(35 - df['temp_max'], 0, 10) * 0.3 +  # Penalty for high temp
        df['fertilizer_usage'] * 0.1
    )
    
    # Add crop type adjustments
    crop_adjustments = {'rice': 1.1, 'maize': 1.0, 'wheat': 0.95, 'barley': 0.9, 'oats': 0.85}
    crop_factors = df['crop_type'].map(crop_adjustments)
    
    # Add noise to make it realistic
    noise = np.random.normal(0, 3, n_samples)
    df['crop_yield'] = (base_yield * crop_factors + noise).clip(lower=5)  # Minimum yield = 5
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("SAMPLE DATA GENERATION COMPLETED")
    print("="*70)
    print(f"✓ Generated {n_samples} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nBasic Statistics:")
    print(df.describe())
    print("="*70 + "\n")
    
    return df


if __name__ == '__main__':
    # Generate sample data with 1000 samples
    generate_sample_data(n_samples=1000, output_path='data/crop_yield_data.csv')
