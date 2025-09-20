"""
Neural network models for materials composition-property modeling.

This module implements forward and inverse neural network architectures for
materials science applications. The forward model predicts material properties
from compositions, while the inverse model generates compositions from target
properties using a generative approach with noise injection.

Classes:
    ForwardNet: Multi-layer perceptron for composition → properties mapping
    InverseNet: Generative MLP for properties → compositions with simplex constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardNet(nn.Module):
    """
    Multi-layer perceptron for forward modeling: composition → properties.
    
    Predicts material properties from material compositions using a simple
    feedforward architecture with ReLU activations. Suitable for multi-output
    regression tasks in materials science.
    
    Args:
        in_dim (int): Input dimension (number of materials/components)
        out_dim (int): Output dimension (number of properties to predict)
        hidden (int, optional): Hidden layer size. Defaults to 128.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        """Initialize the forward network architecture."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input compositions of shape (batch_size, in_dim)
            
        Returns:
            torch.Tensor: Predicted properties of shape (batch_size, out_dim)
        """
        return self.net(x)


class InverseNet(nn.Module):
    """
    Generative MLP for inverse modeling: properties → compositions.
    
    Generates material compositions from target properties using noise injection
    for diversity. Outputs are constrained to the probability simplex (non-negative,
    sum to 1) using softmax activation, ensuring valid material compositions.
    
    Args:
        prop_dim (int): Input property dimension
        out_materials (int): Number of materials/components in output composition
        noise_dim (int, optional): Dimension of noise vector for diversity. Defaults to 16.
        hidden (int, optional): Hidden layer size. Defaults to 128.
        
    Attributes:
        noise_dim (int): Dimension of noise vector used for generation diversity
    """
    
    def __init__(self, prop_dim: int, out_materials: int, noise_dim: int = 16, hidden: int = 128):
        """Initialize the inverse network architecture."""
        super().__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Sequential(
            nn.Linear(prop_dim + noise_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_materials)
        )

    def forward(self, y_norm, z=None):
        """
        Generate compositions from normalized properties and optional noise.
        
        Args:
            y_norm (torch.Tensor): Normalized target properties of shape (batch_size, prop_dim)
            z (torch.Tensor, optional): Noise vector of shape (batch_size, noise_dim).
                                      If None, random noise is generated.
        
        Returns:
            torch.Tensor: Generated compositions on probability simplex of shape
                         (batch_size, out_materials). Each row sums to 1 and is non-negative.
        """
        if z is None:
            z = torch.randn(y_norm.size(0), self.noise_dim, device=y_norm.device)
        h = torch.cat([y_norm, z], dim=1)
        logits = self.fc(h)
        x_hat = F.softmax(logits, dim=1)  # simplex
        return x_hat