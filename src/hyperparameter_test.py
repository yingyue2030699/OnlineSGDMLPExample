#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Study for Materials Composition-Property Modeling.

This module conducts systematic hyperparameter studies to analyze the performance
of forward and inverse models across different configurations. It tests various
architectures, training parameters, and generates comprehensive comparison plots.

Features:
- Grid search over hyperparameters
- Performance tracking and comparison
- Statistical analysis across multiple runs
- Comprehensive visualization of results
- Automated result export with unique identifiers
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import os
import json
import datetime
import itertools
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils import ZScoreNormalizer, set_seed
from synthetic_data_generator import synthesize_dataset
from models import ForwardNet, InverseNet
from training import train_forward_online, eval_forward, train_inverse_online, stream_minibatches
from output import calculate_performance_metrics, generate_unique_identifier


class HyperparameterStudy:
    """
    Systematic hyperparameter study for materials modeling.
    
    Conducts grid search over specified hyperparameters and tracks performance
    metrics across different configurations. Provides comprehensive analysis
    and visualization of results.
    
    Args:
        base_config: Base configuration dictionary
        param_grid: Dictionary of hyperparameters to vary
        n_runs: Number of independent runs per configuration
        results_dir: Base directory for saving results
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        n_runs: int = 3,
        results_dir: str = "results"
    ):
        self.base_config = base_config
        self.param_grid = param_grid
        self.n_runs = n_runs
        self.results_dir = results_dir
        
        # Generate unique study identifier
        self.study_id = self._generate_study_id()
        self.study_dir = os.path.join(results_dir, f"hyperparam_study_{self.study_id}")
        os.makedirs(self.study_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Results storage
        self.results = []
        self.summary_stats = {}
        
        self.logger.info(f"Hyperparameter study initialized: {self.study_id}")
        self.logger.info(f"Results directory: {self.study_dir}")
    
    def _generate_study_id(self) -> str:
        """Generate unique identifier for the study."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_hash = str(hash(str(sorted(self.param_grid.items()))))[-6:]
        return f"{timestamp}_{param_hash}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the study."""
        logger = logging.getLogger(f"hyperparam_study_{self.study_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        log_file = os.path.join(self.study_dir, "study.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        # Get parameter names and values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        configurations = []
        for combination in itertools.product(*param_values):
            config = self.base_config.copy()
            
            # Update config with current parameter combination
            for param_name, param_value in zip(param_names, combination):
                # Handle nested parameters (e.g., "training.forward.lr")
                keys = param_name.split('.')
                current_dict = config
                for key in keys[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                current_dict[keys[-1]] = param_value
            
            configurations.append(config)
        
        return configurations
    
    def _train_single_configuration(
        self,
        config: Dict[str, Any],
        run_id: int,
        config_id: int
    ) -> Dict[str, Any]:
        """Train models with a single configuration."""
        self.logger.info(f"Training configuration {config_id}, run {run_id}")
        
        # Set random seed for reproducibility
        seed = config["random_seed"] + run_id * 1000 + config_id
        set_seed(seed)
        
        # Generate dataset
        X, Y, material_names, property_names = synthesize_dataset(
            n_samples=config["dataset"]["n_samples"],
            n_materials=config["dataset"]["n_materials"],
            n_properties=config["dataset"]["n_properties"],
            alpha_dirichlet=config["dataset"]["alpha_dirichlet"],
            noise_level=config["dataset"]["noise_level"]
        )
        
        # Split data
        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        train_size = int(config["dataset"]["train_split"] * N)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        
        Xtr, Ytr = X[train_idx], Y[train_idx]
        Xte, Yte = X[test_idx], Y[test_idx]
        
        # Normalize properties
        scaler = ZScoreNormalizer()
        scaler.fit(Ytr)
        Ytr_n = scaler.transform(Ytr)
        Yte_n = scaler.transform(Yte)
        
        # Create models
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
        
        # Train forward model
        forward_losses = self._train_forward_with_tracking(fwd, Xtr, Ytr_n, config)
        
        # Evaluate forward model
        train_mse = eval_forward(fwd, Xtr, Ytr_n)
        test_mse = eval_forward(fwd, Xte, Yte_n)
        
        # Generate predictions for detailed metrics
        Y_pred_train = self._forward_predict(fwd, Xtr, scaler)
        Y_pred_test = self._forward_predict(fwd, Xte, scaler)
        
        # Calculate detailed performance metrics
        performance_metrics = calculate_performance_metrics(
            Ytr, Y_pred_train, Yte, Y_pred_test, property_names
        )
        
        # Train inverse model
        inverse_losses, inverse_entropies = self._train_inverse_with_tracking(
            inv, fwd, Ytr_n, config
        )
        
        # Test inverse design
        y_target = np.array(config["inverse_design"]["target_properties"])
        inverse_performance = self._evaluate_inverse_design(
            inv, fwd, y_target, scaler, config["inverse_design"]["n_candidates"]
        )
        
        # Compile results
        result = {
            "config_id": config_id,
            "run_id": run_id,
            "seed": seed,
            "config": config,
            "forward_training": {
                "losses": forward_losses,
                "final_train_mse": train_mse,
                "final_test_mse": test_mse,
                "epochs": len(forward_losses)
            },
            "inverse_training": {
                "consistency_losses": inverse_losses,
                "entropies": inverse_entropies,
                "final_consistency_loss": inverse_losses[-1] if inverse_losses else None,
                "final_entropy": inverse_entropies[-1] if inverse_entropies else None,
                "steps": len(inverse_losses)
            },
            "performance_metrics": performance_metrics,
            "inverse_performance": inverse_performance,
            "model_info": {
                "forward_params": sum(p.numel() for p in fwd.parameters()),
                "inverse_params": sum(p.numel() for p in inv.parameters())
            }
        }
        
        return result
    
    def _train_forward_with_tracking(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        Y_train_norm: np.ndarray,
        config: Dict[str, Any]
    ) -> List[float]:
        """Train forward model with loss tracking."""
        model.train()
        opt = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["forward"]["lr"],
            momentum=config["training"]["forward"]["momentum"],
            weight_decay=config["training"]["forward"]["weight_decay"]
        )
        loss_fn = nn.MSELoss()
        device = next(model.parameters()).device
        
        loss_history = []
        epochs = config["training"]["forward"]["epochs"]
        batch_size = config["training"]["forward"]["batch_size"]
        
        for ep in range(1, epochs + 1):
            running_loss = 0.0
            n_samples = 0
            
            for xb, yb in stream_minibatches(X_train, Y_train_norm, batch_size):
                xb_t = torch.from_numpy(xb).to(device)
                yb_t = torch.from_numpy(yb).to(device)
                
                opt.zero_grad()
                pred = model(xb_t)
                loss = loss_fn(pred, yb_t)
                loss.backward()
                opt.step()
                
                running_loss += loss.item() * xb_t.size(0)
                n_samples += xb_t.size(0)
            
            epoch_loss = running_loss / max(n_samples, 1)
            loss_history.append(epoch_loss)
        
        return loss_history
    
    def _train_inverse_with_tracking(
        self,
        inv_model: nn.Module,
        frozen_forward: nn.Module,
        Y_train_norm: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Train inverse model with loss tracking."""
        inv_model.train()
        frozen_forward.eval()
        
        # Freeze forward model
        for p in frozen_forward.parameters():
            p.requires_grad_(False)
        
        opt = torch.optim.SGD(
            inv_model.parameters(),
            lr=config["training"]["inverse"]["lr"],
            momentum=config["training"]["inverse"]["momentum"],
            weight_decay=config["training"]["inverse"]["l2_logits_weight"]
        )
        
        device = next(inv_model.parameters()).device
        loss_fn = nn.MSELoss()
        
        consistency_history = []
        entropy_history = []
        
        steps = config["training"]["inverse"]["steps"]
        batch_size = config["training"]["inverse"]["batch_size"]
        entropy_weight = config["training"]["inverse"]["entropy_weight"]
        
        N = Y_train_norm.shape[0]
        
        for step in range(1, steps + 1):
            idx = np.random.randint(0, N, size=(batch_size,))
            yb = Y_train_norm[idx]
            yb_t = torch.from_numpy(yb).to(device)
            
            # Generate compositions
            x_hat = inv_model(yb_t)
            
            # Forward-consistency loss
            y_hat = frozen_forward(x_hat)
            loss_consistency = loss_fn(y_hat, yb_t)
            
            # Entropy regularization
            entropy = -(x_hat * (x_hat.clamp_min(1e-9)).log()).sum(dim=1).mean()
            loss = loss_consistency - entropy_weight * entropy
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            consistency_history.append(loss_consistency.item())
            entropy_history.append(entropy.item())
        
        # Unfreeze forward model
        for p in frozen_forward.parameters():
            p.requires_grad_(True)
        
        return consistency_history, entropy_history
    
    def _forward_predict(self, model: nn.Module, x: np.ndarray, scaler: ZScoreNormalizer) -> np.ndarray:
        """Make forward predictions."""
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).to(device)
            y_hat_norm = model(xt).cpu().numpy()
            return scaler.inverse_transform(y_hat_norm)
    
    def _evaluate_inverse_design(
        self,
        inv_model: nn.Module,
        fwd_model: nn.Module,
        y_target: np.ndarray,
        scaler: ZScoreNormalizer,
        n_samples: int
    ) -> Dict[str, float]:
        """Evaluate inverse design performance."""
        inv_model.eval()
        fwd_model.eval()
        device = next(inv_model.parameters()).device
        
        with torch.no_grad():
            # Generate candidates
            y_t_norm = scaler.transform(y_target.reshape(1, -1)).astype(np.float32)
            y_t = torch.from_numpy(np.repeat(y_t_norm, n_samples, axis=0)).to(device)
            x_hat = inv_model(y_t)
            
            # Predict properties
            y_pred_norm = fwd_model(x_hat)
            y_pred = scaler.inverse_transform(y_pred_norm.cpu().numpy())
            
            # Calculate errors
            errors = np.mean((y_pred - y_target) ** 2, axis=1)
            
            return {
                "mean_mse": float(np.mean(errors)),
                "min_mse": float(np.min(errors)),
                "std_mse": float(np.std(errors)),
                "success_rate_01": float(np.mean(errors < 0.1)),  # Success rate with MSE < 0.1
                "success_rate_1": float(np.mean(errors < 1.0)),   # Success rate with MSE < 1.0
            }
    
    def run_study(self):
        """Run the complete hyperparameter study."""
        configurations = self._generate_configurations()
        total_experiments = len(configurations) * self.n_runs
        
        self.logger.info(f"Starting hyperparameter study with {len(configurations)} configurations")
        self.logger.info(f"Total experiments: {total_experiments} ({self.n_runs} runs each)")
        
        experiment_count = 0
        
        for config_id, config in enumerate(configurations):
            self.logger.info(f"Configuration {config_id + 1}/{len(configurations)}")
            
            # Log current configuration
            config_summary = self._summarize_config(config)
            self.logger.info(f"Config: {config_summary}")
            
            for run_id in range(self.n_runs):
                experiment_count += 1
                self.logger.info(f"Experiment {experiment_count}/{total_experiments}")
                
                try:
                    result = self._train_single_configuration(config, run_id, config_id)
                    self.results.append(result)
                    
                    # Log key metrics
                    test_mse = result["forward_training"]["final_test_mse"]
                    inv_mse = result["inverse_performance"]["mean_mse"]
                    self.logger.info(f"Forward test MSE: {test_mse:.6f}, Inverse mean MSE: {inv_mse:.6f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in experiment {experiment_count}: {e}")
                    continue
        
        self.logger.info("Study completed. Analyzing results...")
        self._analyze_results()
        self._save_results()
        self._generate_visualizations()
    
    def _summarize_config(self, config: Dict[str, Any]) -> str:
        """Create a summary string for a configuration."""
        summary_parts = []
        
        # Extract key parameters that are likely to vary
        if "training" in config and "forward" in config["training"]:
            fwd = config["training"]["forward"]
            summary_parts.append(f"fwd_lr={fwd.get('lr', 'N/A')}")
            summary_parts.append(f"fwd_epochs={fwd.get('epochs', 'N/A')}")
        
        if "model_architecture" in config and "forward" in config["model_architecture"]:
            arch = config["model_architecture"]["forward"]
            summary_parts.append(f"fwd_hidden={arch.get('hidden_size', 'N/A')}")
        
        if "training" in config and "inverse" in config["training"]:
            inv = config["training"]["inverse"]
            summary_parts.append(f"inv_lr={inv.get('lr', 'N/A')}")
            summary_parts.append(f"inv_steps={inv.get('steps', 'N/A')}")
        
        return ", ".join(summary_parts)
    
    def _analyze_results(self):
        """Analyze results and compute summary statistics."""
        if not self.results:
            self.logger.warning("No results to analyze")
            return
        
        # Group results by configuration
        config_results = defaultdict(list)
        for result in self.results:
            config_id = result["config_id"]
            config_results[config_id].append(result)
        
        # Compute summary statistics for each configuration
        self.summary_stats = {}
        
        for config_id, results in config_results.items():
            if not results:
                continue
            
            # Extract metrics
            forward_test_mse = [r["forward_training"]["final_test_mse"] for r in results]
            inverse_mean_mse = [r["inverse_performance"]["mean_mse"] for r in results]
            inverse_min_mse = [r["inverse_performance"]["min_mse"] for r in results]
            success_rate_1 = [r["inverse_performance"]["success_rate_1"] for r in results]
            
            # Compute statistics
            self.summary_stats[config_id] = {
                "config": results[0]["config"],
                "n_runs": len(results),
                "forward_test_mse": {
                    "mean": np.mean(forward_test_mse),
                    "std": np.std(forward_test_mse),
                    "min": np.min(forward_test_mse),
                    "max": np.max(forward_test_mse)
                },
                "inverse_mean_mse": {
                    "mean": np.mean(inverse_mean_mse),
                    "std": np.std(inverse_mean_mse),
                    "min": np.min(inverse_mean_mse),
                    "max": np.max(inverse_mean_mse)
                },
                "inverse_min_mse": {
                    "mean": np.mean(inverse_min_mse),
                    "std": np.std(inverse_min_mse),
                    "min": np.min(inverse_min_mse),
                    "max": np.max(inverse_min_mse)
                },
                "inverse_success_rate": {
                    "mean": np.mean(success_rate_1),
                    "std": np.std(success_rate_1),
                    "min": np.min(success_rate_1),
                    "max": np.max(success_rate_1)
                }
            }
        
        # Find best configurations
        best_forward = min(self.summary_stats.items(), 
                          key=lambda x: x[1]["forward_test_mse"]["mean"])
        best_inverse = min(self.summary_stats.items(), 
                          key=lambda x: x[1]["inverse_mean_mse"]["mean"])
        
        self.logger.info(f"Best forward config (ID {best_forward[0]}): "
                        f"MSE = {best_forward[1]['forward_test_mse']['mean']:.6f}")
        self.logger.info(f"Best inverse config (ID {best_inverse[0]}): "
                        f"MSE = {best_inverse[1]['inverse_mean_mse']['mean']:.6f}")
    
    def _save_results(self):
        """Save results to files."""
        # Save raw results
        results_file = os.path.join(self.study_dir, "raw_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary statistics
        summary_file = os.path.join(self.study_dir, "summary_statistics.json")
        with open(summary_file, 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        
        # Save study metadata
        metadata = {
            "study_id": self.study_id,
            "base_config": self.base_config,
            "param_grid": self.param_grid,
            "n_runs": self.n_runs,
            "total_configurations": len(self.summary_stats),
            "total_experiments": len(self.results),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.study_dir, "study_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.study_dir}")
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations of the study results."""
        if not self.summary_stats:
            self.logger.warning("No summary statistics available for visualization")
            return
        
        plots_dir = os.path.join(self.study_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Performance comparison plots
        self._plot_performance_comparison(plots_dir)
        
        # 2. Parameter sensitivity analysis
        self._plot_parameter_sensitivity(plots_dir)
        
        # 3. Training curves for best configurations
        self._plot_best_training_curves(plots_dir)
        
        # 4. Correlation analysis
        self._plot_correlation_analysis(plots_dir)
        
        self.logger.info(f"Visualizations saved to {plots_dir}")
    
    def _plot_performance_comparison(self, plots_dir: str):
        """Plot performance comparison across configurations."""
        config_ids = list(self.summary_stats.keys())
        
        # Extract metrics
        forward_mse_means = [self.summary_stats[cid]["forward_test_mse"]["mean"] for cid in config_ids]
        forward_mse_stds = [self.summary_stats[cid]["forward_test_mse"]["std"] for cid in config_ids]
        inverse_mse_means = [self.summary_stats[cid]["inverse_mean_mse"]["mean"] for cid in config_ids]
        inverse_mse_stds = [self.summary_stats[cid]["inverse_mean_mse"]["std"] for cid in config_ids]
        success_rates = [self.summary_stats[cid]["inverse_success_rate"]["mean"] for cid in config_ids]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Forward model performance
        axes[0, 0].errorbar(config_ids, forward_mse_means, yerr=forward_mse_stds, 
                           fmt='o-', capsize=5, capthick=2)
        axes[0, 0].set_xlabel('Configuration ID')
        axes[0, 0].set_ylabel('Forward Test MSE')
        axes[0, 0].set_title('Forward Model Performance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Inverse model performance
        axes[0, 1].errorbar(config_ids, inverse_mse_means, yerr=inverse_mse_stds, 
                           fmt='s-', capsize=5, capthick=2, color='red')
        axes[0, 1].set_xlabel('Configuration ID')
        axes[0, 1].set_ylabel('Inverse Mean MSE')
        axes[0, 1].set_title('Inverse Model Performance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        axes[1, 0].bar(config_ids, success_rates, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Configuration ID')
        axes[1, 0].set_ylabel('Success Rate (MSE < 1.0)')
        axes[1, 0].set_title('Inverse Design Success Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined performance (forward vs inverse)
        axes[1, 1].scatter(forward_mse_means, inverse_mse_means, s=100, alpha=0.7)
        for i, cid in enumerate(config_ids):
            axes[1, 1].annotate(str(cid), (forward_mse_means[i], inverse_mse_means[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Forward Test MSE')
        axes[1, 1].set_ylabel('Inverse Mean MSE')
        axes[1, 1].set_title('Forward vs Inverse Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, plots_dir: str):
        """Plot parameter sensitivity analysis."""
        # Extract parameter values and performance metrics
        param_data = defaultdict(list)
        performance_data = defaultdict(list)
        
        for config_id, stats in self.summary_stats.items():
            config = stats["config"]
            
            # Extract key parameters
            if "training" in config and "forward" in config["training"]:
                fwd = config["training"]["forward"]
                param_data["forward_lr"].append(fwd.get("lr", None))
                param_data["forward_epochs"].append(fwd.get("epochs", None))
                param_data["forward_batch_size"].append(fwd.get("batch_size", None))
            
            if "model_architecture" in config and "forward" in config["model_architecture"]:
                arch = config["model_architecture"]["forward"]
                param_data["forward_hidden"].append(arch.get("hidden_size", None))
            
            if "training" in config and "inverse" in config["training"]:
                inv = config["training"]["inverse"]
                param_data["inverse_lr"].append(inv.get("lr", None))
                param_data["inverse_steps"].append(inv.get("steps", None))
            
            # Performance metrics
            performance_data["forward_mse"].append(stats["forward_test_mse"]["mean"])
            performance_data["inverse_mse"].append(stats["inverse_mean_mse"]["mean"])
        
        # Create sensitivity plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for param_name, param_values in param_data.items():
            if plot_idx >= len(axes):
                break
            
            if None in param_values or len(set(param_values)) <= 1:
                continue  # Skip parameters that don't vary
            
            # Plot forward performance vs parameter
            axes[plot_idx].scatter(param_values, performance_data["forward_mse"], 
                                 alpha=0.7, label='Forward MSE', s=60)
            axes[plot_idx].scatter(param_values, performance_data["inverse_mse"], 
                                 alpha=0.7, label='Inverse MSE', s=60)
            
            axes[plot_idx].set_xlabel(param_name.replace('_', ' ').title())
            axes[plot_idx].set_ylabel('MSE')
            axes[plot_idx].set_title(f'Performance vs {param_name.replace("_", " ").title()}')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Use log scale for learning rates
            if 'lr' in param_name:
                axes[plot_idx].set_xscale('log')
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "parameter_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_training_curves(self, plots_dir: str):
        """Plot training curves for best configurations."""
        # Find best configurations
        best_forward_id = min(self.summary_stats.items(), 
                             key=lambda x: x[1]["forward_test_mse"]["mean"])[0]
        best_inverse_id = min(self.summary_stats.items(), 
                             key=lambda x: x[1]["inverse_mean_mse"]["mean"])[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get training curves for best configurations
        for result in self.results:
            if result["config_id"] == best_forward_id:
                forward_losses = result["forward_training"]["losses"]
                axes[0, 0].plot(forward_losses, alpha=0.7, linewidth=1)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title(f'Best Forward Config (ID {best_forward_id}) - Training Curves')
        axes[ 0, 0].grid(True, alpha=0.3)
        
        for result in self.results:
            if result["config_id"] == best_inverse_id:
                inverse_losses = result["inverse_training"]["consistency_losses"]
                if inverse_losses:
                    # Subsample for cleaner visualization
                    step_indices = range(0, len(inverse_losses), max(1, len(inverse_losses) // 100))
                    subsampled_losses = [inverse_losses[i] for i in step_indices]
                    axes[0, 1].plot(step_indices, subsampled_losses, alpha=0.7, linewidth=1)
        
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Consistency Loss')
        axes[0, 1].set_title(f'Best Inverse Config (ID {best_inverse_id}) - Consistency Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        for result in self.results:
            if result["config_id"] == best_inverse_id:
                entropies = result["inverse_training"]["entropies"]
                if entropies:
                    # Subsample for cleaner visualization
                    step_indices = range(0, len(entropies), max(1, len(entropies) // 100))
                    subsampled_entropies = [entropies[i] for i in step_indices]
                    axes[1, 0].plot(step_indices, subsampled_entropies, alpha=0.7, linewidth=1)
        
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title(f'Best Inverse Config (ID {best_inverse_id}) - Entropy Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance distribution for best configs
        best_forward_results = [r for r in self.results if r["config_id"] == best_forward_id]
        best_inverse_results = [r for r in self.results if r["config_id"] == best_inverse_id]
        
        forward_test_mse = [r["forward_training"]["final_test_mse"] for r in best_forward_results]
        inverse_mean_mse = [r["inverse_performance"]["mean_mse"] for r in best_inverse_results]
        
        axes[1, 1].hist(forward_test_mse, alpha=0.7, label=f'Forward (ID {best_forward_id})', bins=10)
        axes[1, 1].hist(inverse_mean_mse, alpha=0.7, label=f'Inverse (ID {best_inverse_id})', bins=10)
        axes[1, 1].set_xlabel('Final MSE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Performance Distribution - Best Configs')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "best_training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, plots_dir: str):
        """Plot correlation analysis between different metrics."""
        # Extract all metrics
        metrics_data = {
            'forward_test_mse': [],
            'inverse_mean_mse': [],
            'inverse_min_mse': [],
            'success_rate': [],
            'forward_params': [],
            'inverse_params': []
        }
        
        for result in self.results:
            metrics_data['forward_test_mse'].append(result["forward_training"]["final_test_mse"])
            metrics_data['inverse_mean_mse'].append(result["inverse_performance"]["mean_mse"])
            metrics_data['inverse_min_mse'].append(result["inverse_performance"]["min_mse"])
            metrics_data['success_rate'].append(result["inverse_performance"]["success_rate_1"])
            metrics_data['forward_params'].append(result["model_info"]["forward_params"])
            metrics_data['inverse_params'].append(result["model_info"]["inverse_params"])
        
        # Create correlation matrix
        import pandas as pd
        df = pd.DataFrame(metrics_data)
        correlation_matrix = df.corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[0])
        axes[0].set_title('Metrics Correlation Matrix')
        
        # Scatter plot of key relationships
        axes[1].scatter(metrics_data['forward_test_mse'], metrics_data['inverse_mean_mse'], 
                       alpha=0.6, s=60)
        axes[1].set_xlabel('Forward Test MSE')
        axes[1].set_ylabel('Inverse Mean MSE')
        axes[1].set_title('Forward vs Inverse Performance')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(metrics_data['forward_test_mse'], metrics_data['inverse_mean_mse'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(metrics_data['forward_test_mse']), 
                             max(metrics_data['forward_test_mse']), 100)
        axes[1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "correlation_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to run hyperparameter study.
    
    Defines the base configuration and parameter grid, then runs
    a comprehensive study with multiple random seeds.
    """
    
    # Base configuration
    base_config = {
        "random_seed": 42,
        "dataset": {
            "n_samples": 2000,  # Smaller for faster experiments
            "n_materials": 8,
            "n_properties": 3,
            "alpha_dirichlet": 1.3,
            "noise_level": 0.01,
            "train_split": 0.8
        },
        "model_architecture": {
            "forward": {
                "hidden_size": 64,
                "activation": "ReLU"
            },
            "inverse": {
                "hidden_size": 64,
                "noise_dim": 16,
                "activation": "ReLU"
            }
        },
        "training": {
            "forward": {
                "lr": 0.01,
                "epochs": 5,
                "batch_size": 32,
                "weight_decay": 1e-4,
                "optimizer": "SGD",
                "momentum": 0.9
            },
            "inverse": {
                "lr": 0.005,
                "steps": 1000,
                "batch_size": 32,
                "entropy_weight": 5e-3,
                "l2_logits_weight": 1e-4,
                "optimizer": "SGD",
                "momentum": 0.9
            }
        },
        "inverse_design": {
            "target_properties": [60.0, 1.5, 0.7],
            "n_candidates": 20
        }
    }
    
    # Parameter grid to search over
    param_grid = {
        "training.forward.lr": [0.005],
        "training.forward.epochs": [3, 5, 8, 12],
        "model_architecture.forward.hidden_size": [32, 64, 128, 256],
        "training.inverse.lr": [0.001],
        "training.inverse.steps": [500],
        "training.inverse.entropy_weight": [1e-3]
    }
    
    # Run study
    study = HyperparameterStudy(
        base_config=base_config,
        param_grid=param_grid,
        n_runs=3,  # 3 independent runs per configuration
        results_dir="results"
    )
    
    study.run_study()
    
    print(f"\nHyperparameter study completed!")
    print(f"Results saved to: {study.study_dir}")
    print(f"Study ID: {study.study_id}")


if __name__ == "__main__":
    main()