"""
Training utilities for online SGD-based materials modeling.

This module provides training functions for both forward and inverse neural networks
using online/mini-batch stochastic gradient descent. Includes specialized training
for inverse models with consistency losses and regularization terms.

Functions:
    stream_minibatches: Iterator for streaming mini-batches with shuffling
    train_forward_online: Train forward model with online SGD
    eval_forward: Evaluate forward model performance
    train_inverse_online: Train inverse model with consistency and entropy regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stream_minibatches(X, Y, batch_size: int):
    """
    Simple streaming iterator that shuffles data and yields mini-batches.
    
    Provides an iterator over the dataset that shuffles the data order
    each time it's called and yields mini-batches of the specified size.
    
    Args:
        X (np.ndarray): Input data of shape (N, ...)
        Y (np.ndarray): Target data of shape (N, ...)
        batch_size (int): Size of each mini-batch
        
    Yields:
        Tuple[np.ndarray, np.ndarray]: Mini-batch of (X_batch, Y_batch)
    """
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        j = idx[start:start+batch_size]
        yield X[j], Y[j]


def train_forward_online(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train_norm: np.ndarray,
    lr: float = 1e-2,
    epochs: int = 5,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    verbose: bool = True,
):
    """
    Train forward model using online SGD with mini-batches.
    
    Trains a forward neural network to predict normalized properties from
    compositions using stochastic gradient descent with momentum and weight decay.
    Data is shuffled each epoch and processed in mini-batches.
    
    Args:
        model (nn.Module): Forward neural network to train
        X_train (np.ndarray): Training compositions of shape (N, M)
        Y_train_norm (np.ndarray): Normalized training properties of shape (N, P)
        lr (float, optional): Learning rate. Defaults to 1e-2.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Mini-batch size. Use 1 for true online learning. Defaults to 32.
        weight_decay (float, optional): L2 regularization strength. Defaults to 1e-4.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device

    for ep in range(1, epochs+1):
        running = 0.0
        n = 0
        for xb, yb in stream_minibatches(X_train, Y_train_norm, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            opt.zero_grad()
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            opt.step()

            running += loss.item() * xb_t.size(0)
            n += xb_t.size(0)
        if verbose:
            print(f"[Forward] epoch {ep}/{epochs}  loss={running/max(n,1):.6f}")


@torch.no_grad()
def eval_forward(model: nn.Module, X: np.ndarray, Y_norm: np.ndarray) -> float:
    """
    Evaluate forward model performance using MSE loss.
    
    Computes mean squared error between model predictions and ground truth
    on normalized property space. Model is set to evaluation mode during inference.
    
    Args:
        model (nn.Module): Trained forward model
        X (np.ndarray): Test compositions of shape (N, M)
        Y_norm (np.ndarray): Normalized test properties of shape (N, P)
        
    Returns:
        float: Mean squared error on the test set
    """
    model.eval()
    device = next(model.parameters()).device
    xb = torch.from_numpy(X).to(device)
    yb = torch.from_numpy(Y_norm).to(device)
    pred = model(xb)
    return F.mse_loss(pred, yb).item()


def train_inverse_online(
    inv_model: nn.Module,
    frozen_forward: nn.Module,
    Y_train_norm: np.ndarray,
    lr: float = 5e-3,
    steps: int = 5000,
    batch_size: int = 64,
    entropy_weight: float = 1e-2,
    l2_logits_weight: float = 1e-4,
    verbose_every: int = 500,
):
    """
    Train inverse model with forward consistency and entropy regularization.
    
    Trains an inverse neural network to generate compositions from properties
    by minimizing the reconstruction error when the generated compositions
    are passed through a frozen forward model. Includes entropy regularization
    to encourage diverse compositions and L2 regularization on pre-softmax logits.
    
    The loss function combines:
    - Forward consistency: MSE between target and forward(inverse(target))
    - Entropy regularization: Encourages diverse/spread compositions
    - L2 regularization: Applied via weight decay on model parameters
    
    Args:
        inv_model (nn.Module): Inverse model to train
        frozen_forward (nn.Module): Pre-trained forward model (will be frozen during training)
        Y_train_norm (np.ndarray): Normalized training properties of shape (N, P)
        lr (float, optional): Learning rate. Defaults to 5e-3.
        steps (int, optional): Number of training steps. Defaults to 5000.
        batch_size (int, optional): Mini-batch size. Defaults to 64.
        entropy_weight (float, optional): Weight for entropy regularization term. Defaults to 1e-2.
        l2_logits_weight (float, optional): Weight decay for L2 regularization. Defaults to 1e-4.
        verbose_every (int, optional): Print progress every N steps. Set to None to disable. Defaults to 500.
        
    Note:
        The forward model parameters are temporarily frozen during training and
        unfrozen at the end in case the caller wants to fine-tune later.
    """
    inv_model.train()
    frozen_forward.eval()
    for p in frozen_forward.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(inv_model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_logits_weight)
    device = next(inv_model.parameters()).device
    loss_fn = nn.MSELoss()

    N = Y_train_norm.shape[0]
    for step in range(1, steps+1):
        idx = np.random.randint(0, N, size=(batch_size,))
        yb = Y_train_norm[idx]
        yb_t = torch.from_numpy(yb).to(device)

        # Generate compositions
        x_hat = inv_model(yb_t)  # (B, M), simplex

        # Forward-consistency loss
        y_hat = frozen_forward(x_hat)
        loss_consistency = loss_fn(y_hat, yb_t)

        # Entropy regularization (maximize entropy -> minimize negative entropy)
        # entropy = -sum x log x
        entropy = - (x_hat * (x_hat.clamp_min(1e-9)).log()).sum(dim=1).mean()
        loss = loss_consistency - entropy_weight * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose_every and step % verbose_every == 0:
            print(f"[Inverse] step {step}/{steps} "
                  f"consistency={loss_consistency.item():.6f}  H={entropy.item():.4f}")

    # Unfreeze in case caller wants to fine-tune:
    for p in frozen_forward.parameters():
        p.requires_grad_(True)
