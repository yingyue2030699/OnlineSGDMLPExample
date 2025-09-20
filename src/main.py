#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online SGD MLPs for Materials Composition-Property Modeling.

Streamlined main execution file for materials science machine learning pipeline
with comprehensive logging and organized output management.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import logging

from utils import ZScoreNormalizer, set_seed
from synthetic_data_generator import synthesize_dataset
from models import ForwardNet, InverseNet
from training import *
from output import ExperimentLogger, calculate_performance_metrics


def composition_to_dict(x: np.ndarray, materials: List[str], cutoff: float = 0.01) -> Dict[str, float]:
    """Convert composition array to dictionary with material names and fractions."""
    return {mat: float(frac) for mat, frac in zip(materials, x) if frac >= cutoff}


@torch.no_grad()
def forward_predict(model: nn.Module, x: np.ndarray, y_norm: ZScoreNormalizer) -> np.ndarray:
    """Make forward predictions and denormalize to original property scale."""
    device = next(model.parameters()).device
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    y_hat_norm = model(xt).cpu().numpy()
    return y_norm.inverse_transform(y_hat_norm)


@torch.no_grad()
def inverse_design(
    inv_model: nn.Module,
    fwd_model: nn.Module,
    y_target_raw: np.ndarray,
    y_norm: ZScoreNormalizer,
    n_samples: int = 10,
) -> List[np.ndarray]:
    """Generate multiple candidate compositions for target properties."""
    device = next(inv_model.parameters()).device
    y_t_norm = y_norm.transform(y_target_raw.reshape(1, -1)).astype(np.float32)
    y_t = torch.from_numpy(np.repeat(y_t_norm, n_samples, axis=0)).to(device)
    x_hat = inv_model(y_t)  # (n_samples, M)
    return [x_hat[i].cpu().numpy() for i in range(n_samples)]


def train_forward_online_with_logging(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train_norm: np.ndarray,
    logger: logging.Logger,
    lr: float = 1e-2,
    epochs: int = 5,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
) -> List[float]:
    """Train forward model and return loss history."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device

    loss_history = []
    logger.info("Starting forward model training...")

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

        epoch_loss = running / max(n, 1)
        loss_history.append(epoch_loss)
        logger.info(
            f"Forward training epoch {ep}/{epochs}: loss={epoch_loss:.6f}")

    return loss_history


def train_inverse_online_with_logging(
    inv_model: nn.Module,
    frozen_forward: nn.Module,
    Y_train_norm: np.ndarray,
    logger: logging.Logger,
    lr: float = 5e-3,
    steps: int = 5000,
    batch_size: int = 64,
    entropy_weight: float = 1e-2,
    l2_logits_weight: float = 1e-4,
    log_every: int = 500,
) -> Tuple[List[float], List[float]]:
    """Train inverse model and return loss histories."""
    inv_model.train()
    frozen_forward.eval()
    for p in frozen_forward.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(inv_model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=l2_logits_weight)
    device = next(inv_model.parameters()).device
    loss_fn = nn.MSELoss()

    consistency_history = []
    entropy_history = []

    logger.info("Starting inverse model training...")

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

        # Entropy regularization
        entropy = - (x_hat * (x_hat.clamp_min(1e-9)).log()).sum(dim=1).mean()
        loss = loss_consistency - entropy_weight * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Log every step for plotting
        consistency_history.append(loss_consistency.item())
        entropy_history.append(entropy.item())

        if step % log_every == 0:
            logger.info(f"Inverse training step {step}/{steps}: "
                        f"consistency={loss_consistency.item():.6f}, entropy={entropy.item():.4f}")

    # Unfreeze in case caller wants to fine-tune
    for p in frozen_forward.parameters():
        p.requires_grad_(True)

    return consistency_history, entropy_history


def main():
    """Main experiment execution with comprehensive logging and output management."""
    # Experiment configuration
    config = {
        "random_seed": 42,
        "dataset": {
            "n_samples": 6000,
            "n_materials": 12,
            "n_properties": 3,
            "alpha_dirichlet": 1.3,
            "noise_level": 0.01,
            "train_split": 0.8
        },
        "model_architecture": {
            "forward": {
                "hidden_size": 128,
                "activation": "ReLU"
            },
            "inverse": {
                "hidden_size": 128,
                "noise_dim": 16,
                "activation": "ReLU"
            }
        },
        "training": {
            "forward": {
                "lr": 0.02,
                "epochs": 6,
                "batch_size": 32,
                "weight_decay": 1e-4,
                "optimizer": "SGD",
                "momentum": 0.9
            },
            "inverse": {
                "lr": 0.01,
                "steps": 4000,
                "batch_size": 64,
                "entropy_weight": 8e-3,
                "l2_logits_weight": 1e-4,
                "optimizer": "SGD",
                "momentum": 0.9
            }
        },
        "inverse_design": {
            "target_properties": [80.0, 2.0, 0.7],
            "n_candidates": 50
        }
    }

    # Initialize experiment logger
    exp_logger = ExperimentLogger(config)
    logger = exp_logger.logger

    # Set random seed
    set_seed(config["random_seed"])
    logger.info(f"Random seed set to: {config['random_seed']}")

    # 1) Generate dataset
    logger.info("Generating synthetic dataset...")
    X, Y, material_names, property_names = synthesize_dataset(
        n_samples=config["dataset"]["n_samples"],
        n_materials=config["dataset"]["n_materials"],
        n_properties=config["dataset"]["n_properties"],
        alpha_dirichlet=config["dataset"]["alpha_dirichlet"],
        noise_level=config["dataset"]["noise_level"]
    )

    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    train_size = int(config["dataset"]["train_split"] * N)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    logger.info(
        f"Dataset created: {N} total samples ({len(train_idx)} train, {len(test_idx)} test)")
    logger.info(
        f"Features: {len(material_names)} materials, {len(property_names)} properties")

    # Normalize properties
    scaler = ZScoreNormalizer()
    scaler.fit(Ytr)
    Ytr_n = scaler.transform(Ytr)
    Yte_n = scaler.transform(Yte)
    logger.info("Property normalization applied")

    # 2) Define models
    device = torch.device("cpu")
    fwd = ForwardNet(
        in_dim=X.shape[1],
        out_dim=Y.shape[1],
        hidden=config["model_architecture"]["forward"]["hidden_size"]
    ).to(device)

    inv = InverseNet(
        prop_dim=Y.shape[1],
        out_materials=X.shape[1],
        noise_dim=config["model_architecture"]["inverse"]["noise_dim"],
        hidden=config["model_architecture"]["inverse"]["hidden_size"]
    ).to(device)

    logger.info(f"Models initialized on device: {device}")
    logger.info(
        f"Forward model parameters: {sum(p.numel() for p in fwd.parameters())}")
    logger.info(
        f"Inverse model parameters: {sum(p.numel() for p in inv.parameters())}")

    # 3) Train forward model
    forward_losses = train_forward_online_with_logging(
        fwd, Xtr, Ytr_n, logger,
        lr=config["training"]["forward"]["lr"],
        epochs=config["training"]["forward"]["epochs"],
        batch_size=config["training"]["forward"]["batch_size"],
        weight_decay=config["training"]["forward"]["weight_decay"]
    )

    te_mse = eval_forward(fwd, Xte, Yte_n)
    logger.info(
        f"Forward model training complete. Test MSE (normalized): {te_mse:.6f}")

    # 4) Train inverse model
    inverse_consistency, inverse_entropy = train_inverse_online_with_logging(
        inv_model=inv,
        frozen_forward=fwd,
        Y_train_norm=Ytr_n,
        logger=logger,
        lr=config["training"]["inverse"]["lr"],
        steps=config["training"]["inverse"]["steps"],
        batch_size=config["training"]["inverse"]["batch_size"],
        entropy_weight=config["training"]["inverse"]["entropy_weight"],
        l2_logits_weight=config["training"]["inverse"]["l2_logits_weight"],
        log_every=500
    )
    logger.info("Inverse model training complete")

    # 5) Generate predictions
    logger.info("Generating model predictions...")
    Y_pred_train = forward_predict(fwd, Xtr, scaler)
    Y_pred_test = forward_predict(fwd, Xte, scaler)

    # 6) Inverse design
    logger.info("Performing inverse design...")
    y_target = np.array(config["inverse_design"]["target_properties"])
    candidates = inverse_design(inv, fwd, y_target, scaler,
                                n_samples=config["inverse_design"]["n_candidates"])
    predicted_properties = [forward_predict(fwd, x[None, :], scaler)[
        0] for x in candidates]
    logger.info(f"Generated {len(candidates)} inverse design candidates")

    # 7) Calculate performance metrics
    logger.info("Calculating performance metrics...")
    performance_metrics = calculate_performance_metrics(
        Ytr, Y_pred_train, Yte, Y_pred_test, property_names
    )

    # 8) Prepare and save all results
    logger.info("Preparing experiment metadata...")

    model_info = {
        "forward_model": {
            "type": "ForwardNet",
            "input_dim": int(X.shape[1]),
            "output_dim": int(Y.shape[1]),
            "hidden_size": config["model_architecture"]["forward"]["hidden_size"],
            "activation": config["model_architecture"]["forward"]["activation"],
            "total_parameters": sum(p.numel() for p in fwd.parameters())
        },
        "inverse_model": {
            "type": "InverseNet",
            "property_dim": int(Y.shape[1]),
            "output_materials": int(X.shape[1]),
            "noise_dim": config["model_architecture"]["inverse"]["noise_dim"],
            "hidden_size": config["model_architecture"]["inverse"]["hidden_size"],
            "activation": config["model_architecture"]["inverse"]["activation"],
            "total_parameters": sum(p.numel() for p in inv.parameters())
        }
    }

    data_info = {
        "total_samples": int(N),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "n_materials": int(X.shape[1]),
        "n_properties": int(Y.shape[1]),
        "material_names": material_names,
        "property_names": property_names,
        "property_statistics": {
            "train": {
                "mean": Ytr.mean(axis=0).tolist(),
                "std": Ytr.std(axis=0).tolist(),
                "min": Ytr.min(axis=0).tolist(),
                "max": Ytr.max(axis=0).tolist()
            },
            "test": {
                "mean": Yte.mean(axis=0).tolist(),
                "std": Yte.std(axis=0).tolist(),
                "min": Yte.min(axis=0).tolist(),
                "max": Yte.max(axis=0).tolist()
            }
        }
    }

    training_info = {
        "forward_training": {
            "final_train_loss": forward_losses[-1],
            "final_test_loss": float(te_mse),
            "total_epochs": len(forward_losses)
        },
        "inverse_training": {
            "final_consistency_loss": inverse_consistency[-1],
            "final_entropy": inverse_entropy[-1],
            "total_steps": len(inverse_consistency)
        }
    }

    # Save all results
    logger.info("Saving experiment results...")

    exp_logger.save_experiment_metadata(
        model_info, data_info, performance_metrics, training_info)
    exp_logger.save_model_checkpoints(fwd, inv)
    exp_logger.save_data_artifacts(
        Xtr, Ytr, Xte, Yte, scaler, material_names, property_names)

    training_losses = {
        'forward': forward_losses,
        # Subsample for cleaner plots
        'inverse_consistency': inverse_consistency[::50],
        'inverse_entropy': inverse_entropy[::50]
    }
    exp_logger.save_predictions_and_results(Y_pred_train, Y_pred_test, candidates,
                                            predicted_properties, y_target, training_losses)

    # Generate visualizations
    exp_logger.generate_visualizations(
        Xtr, Ytr, Xte, Yte, Y_pred_train, Y_pred_test,
        candidates, predicted_properties, y_target,
        material_names, property_names, training_losses
    )

    # Log final summary
    exp_logger.log_experiment_summary(
        performance_metrics, candidates, predicted_properties,
        y_target, material_names, property_names
    )


if __name__ == "__main__":
    main()
