"""
Synthetic dataset generation for materials composition-property modeling.

This module generates synthetic materials datasets with realistic composition-property
relationships for testing and development of machine learning models. Compositions
are sampled from Dirichlet distributions to ensure valid probability simplexes,
while properties are computed using nonlinear functions with interaction terms.

Functions:
    synthesize_dataset: Generate synthetic materials dataset with compositions and properties
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def synthesize_dataset(
    n_samples: int = 6000,
    n_materials: int = 12,
    n_properties: int = 3,
    alpha_dirichlet: float = 1.5,
    noise_level: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate synthetic materials dataset with realistic composition-property relationships.
    
    Creates a dataset where compositions are sampled from Dirichlet distributions
    (ensuring valid probability simplexes) and properties are computed using
    nonlinear functions with interaction terms between materials. Includes
    configurable noise for realistic modeling scenarios.
    
    The generated properties have different scales and characteristics:
    - Property 0: Large scale (0-100+) with quadratic interactions
    - Property 1: Medium scale (-3 to +3) with tanh nonlinearity  
    - Property 2: Small scale (0-1) with oscillatory behavior
    
    Args:
        n_samples (int, optional): Number of samples to generate. Defaults to 6000.
        n_materials (int, optional): Number of materials/components. Defaults to 12.
        n_properties (int, optional): Number of properties to compute (max 3). Defaults to 3.
        alpha_dirichlet (float, optional): Dirichlet concentration parameter.
                                         Higher values create more uniform compositions. Defaults to 1.5.
        noise_level (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.02.
    
    Returns:
        Tuple containing:
            - X (np.ndarray): Compositions of shape (n_samples, n_materials).
                            Each row sums to 1 and is non-negative.
            - Y (np.ndarray): Properties of shape (n_samples, n_properties).
            - material_names (List[str]): Names of materials (Mat_000, Mat_001, etc.)
            - property_names (List[str]): Names of properties (Prop_000, Prop_001, etc.)
    
    Example:
        >>> X, Y, mat_names, prop_names = synthesize_dataset(
        ...     n_samples=1000, n_materials=8, n_properties=2
        ... )
        >>> print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
        >>> print(f"Composition sum check: {np.allclose(X.sum(axis=1), 1.0)}")
    """
    mat_names = [f"Mat_{i:03d}" for i in range(n_materials)]
    prop_names = [f"Prop_{i:03d}" for i in range(n_properties)]

    # Dirichlet for compositions
    alpha = np.full(n_materials, alpha_dirichlet, dtype=np.float64)
    X = np.random.dirichlet(alpha, size=n_samples).astype(np.float64)

    # Ground-truth property mapping (nonlinear + interactions)
    # y1 ~ larger scale (0..~100)
    A1 = np.random.uniform(0.5, 2.0, size=(n_materials,))
    M1 = np.random.uniform(-1.0, 1.0, size=(n_materials, n_materials))
    M1 = 0.5 * (M1 + M1.T)  # symmetric interactions
    y1 = 60.0 * (X @ A1) + 20.0 * np.sum((X @ M1) * X, axis=1)
    y1 = np.maximum(0.0, y1)  # keep nonnegative-ish

    # y2 ~ medium range (-3..+3)
    A2 = np.random.uniform(-1.0, 1.0, size=(n_materials,))
    y2 = 3.0 * np.tanh(X @ A2 - 0.2)

    # y3 ~ small [0..1] oscillatory
    v = np.random.uniform(0.0, 1.0, size=(n_materials,))
    y3 = 0.5 + 0.5 * np.sin(4.0 * math.pi * (X @ v))

    Ys = [y1, y2, y3][:n_properties]
    Y = np.vstack(Ys).T

    # Additive noise
    Y += noise_level * np.random.randn(*Y.shape)

    return X.astype(np.float32), Y.astype(np.float32), mat_names, prop_names
