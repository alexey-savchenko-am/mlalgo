import numpy as np
from typing import Tuple
Array = np.ndarray

def get_random_slice(X: Array, y: Array, size: int = 5, seed: int = 42) -> Tuple[Array, Array]:
    """
        Return a random slice of the dataset.
    
    Args:
        X (Array): Features.
        y (Array): Target values.
        size (int): Number of samples to select (default=5).
        seed (int): Random seed for reproducibility (default=42).
    
    Returns:
        Tuple[Array, Array]: Random subset of X and y.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=size, replace=False)
    return X[idx], y[idx]
