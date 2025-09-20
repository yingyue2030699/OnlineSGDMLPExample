"""
Utility functions and classes for materials science machine learning.

This module provides essential utilities for data preprocessing and reproducibility
in materials composition-property modeling, including normalization tools and
random seed management.

Classes:
    ZScoreNormalizer: Per-feature z-score normalization with inverse transform capability

Functions:
    set_seed: Set random seeds for reproducible results across numpy, torch, and random
"""

import random

import numpy as np
import torch

class ZScoreNormalizer:
    """
    Per-feature z-score normalization with inverse transformation capability.
    
    Normalizes data to have zero mean and unit variance per feature, with the ability
    to transform data back to original scale. Handles zero standard deviation by
    setting std to 1.0 to avoid division by zero.
    
    Attributes:
        mean (np.ndarray): Mean values per feature, shape (P,)
        std (np.ndarray): Standard deviation per feature, shape (P,)
    """
    def __init__(self):
        """Initialize the normalizer with empty statistics."""
        self.mean = None
        self.std = None

    def fit(self, y: np.ndarray):
        """
        Compute and store normalization statistics from training data.
        
        Args:
            y (np.ndarray): Training data of shape (N, P) where N is number of
                          samples and P is number of features
        """
        # y: (N, P)
        self.mean = y.mean(axis=0)
        self.std = y.std(axis=0)
        self.std[self.std == 0.0] = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to input data.
        
        Args:
            y (np.ndarray): Input data of shape (N, P)
            
        Returns:
            np.ndarray: Normalized data with zero mean and unit variance per feature
        """
        return (y - self.mean) / self.std

    def inverse_transform(self, y_z: np.ndarray) -> np.ndarray:
        """
        Transform normalized data back to original scale.
        
        Args:
            y_z (np.ndarray): Normalized data of shape (N, P)
            
        Returns:
            np.ndarray: Data transformed back to original scale
        """
        return y_z * self.std + self.mean


def set_seed(seed: int = 123):
    """
    Set random seeds for reproducible results across all random number generators.
    
    Sets seeds for Python's random module, numpy, and PyTorch to ensure
    reproducible results across different runs.
    
    Args:
        seed (int, optional): Random seed value. Defaults to 123.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)