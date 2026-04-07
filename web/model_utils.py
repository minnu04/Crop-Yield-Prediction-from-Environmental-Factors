from pathlib import Path

import joblib


def load_model(filepath: str):
    """Load a saved joblib model from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(path)
