"""
Output and results management for materials modeling experiments.

This module handles all aspects of experiment output including:
- Unique experiment ID generation
- Comprehensive metadata saving
- Model checkpoint management
- Data artifact preservation
- Results visualization and logging

Classes:
    ExperimentLogger: Main class for managing experiment outputs

Functions:
    generate_unique_identifier: Create unique experiment IDs
    calculate_performance_metrics: Compute comprehensive model performance metrics
"""

import os
import json
import datetime
import hashlib
import platform
import sys
import logging
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils import ZScoreNormalizer
from plotting import create_comprehensive_report


def generate_unique_identifier(config: Dict[str, Any]) -> str:
    """
    Generate a unique identifier for the experiment based on configuration and timestamp.
    
    Args:
        config: Dictionary containing experiment configuration
        
    Returns:
        str: Unique identifier string in format YYYYMMDD_HHMMSS_confighash
    """
    # Create a hash of the configuration for reproducibility tracking
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Add timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{timestamp}_{config_hash}"


def calculate_performance_metrics(
    Y_true_train: np.ndarray,
    Y_pred_train: np.ndarray,
    Y_true_test: np.ndarray,
    Y_pred_test: np.ndarray,
    property_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for both training and test sets.
    
    Computes MSE, RMSE, MAE, and R² for overall performance and per-property
    performance on both training and test datasets.
    
    Args:
        Y_true_train, Y_pred_train: Training true and predicted values, shape (N, P)
        Y_true_test, Y_pred_test: Test true and predicted values, shape (M, P)
        property_names: Names of properties
        
    Returns:
        Dict containing nested performance metrics:
        {
            "train": {"overall": {...}, "per_property": {prop_name: {...}}},
            "test": {"overall": {...}, "per_property": {prop_name: {...}}}
        }
    """
    metrics = {}
    
    for split, y_true, y_pred in [("train", Y_true_train, Y_pred_train), 
                                  ("test", Y_true_test, Y_pred_test)]:
        split_metrics = {}
        
        # Overall metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        split_metrics["overall"] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        }
        
        # Per-property metrics
        split_metrics["per_property"] = {}
        for i, prop_name in enumerate(property_names):
            prop_mse = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
            prop_rmse = np.sqrt(prop_mse)
            prop_mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            
            # R-squared
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            split_metrics["per_property"][prop_name] = {
                "mse": float(prop_mse),
                "rmse": float(prop_rmse),
                "mae": float(prop_mae),
                "r2": float(r2)
            }
        
        metrics[split] = split_metrics
    
    return metrics


class ExperimentLogger:
    """
    Comprehensive experiment logging and output management.
    
    Handles all aspects of experiment output including directory creation,
    metadata saving, model checkpoints, data artifacts, and visualization
    generation. Provides a centralized interface for experiment tracking.
    
    Args:
        config: Experiment configuration dictionary
        base_results_dir: Base directory for all results (default: "results")
        
    Attributes:
        config: Experiment configuration
        unique_id: Unique experiment identifier
        result_dir: Full path to experiment results directory
        logger: Configured logger for this experiment
    """
    
    def __init__(self, config: Dict[str, Any], base_results_dir: str = "results"):
        """Initialize experiment logger with unique directory and logging setup."""
        self.config = config
        self.unique_id = generate_unique_identifier(config)
        self.result_dir = os.path.join(base_results_dir, self.unique_id)
        
        # Create result directory structure
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "plots"), exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Experiment initialized with ID: {self.unique_id}")
        self.logger.info(f"Results directory: {self.result_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment-specific logging configuration."""
        logger = logging.getLogger(f"experiment_{self.unique_id}")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # File handler
        log_file = os.path.join(self.result_dir, "experiment.log")
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
    
    def save_experiment_metadata(
        self,
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        training_info: Dict[str, Any]
    ):
        """
        Save comprehensive experiment metadata to JSON file.
        
        Args:
            model_info: Model architecture information
            data_info: Dataset information and statistics
            performance_metrics: Model performance metrics
            training_info: Training process information
        """
        metadata = {
            "experiment_info": {
                "experiment_id": self.unique_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "pytorch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
            "configuration": self.config,
            "model_architecture": model_info,
            "dataset_info": data_info,
            "performance_metrics": performance_metrics,
            "training_info": training_info,
        }
        
        metadata_path = os.path.join(self.result_dir, "experiment_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Experiment metadata saved to: {metadata_path}")
    
    def save_model_checkpoints(self, forward_model: nn.Module, inverse_model: nn.Module):
        """
        Save model checkpoints in multiple formats.
        
        Saves both state dictionaries (for loading into same architecture) and
        complete models (including architecture) for maximum flexibility.
        
        Args:
            forward_model: Trained forward model
            inverse_model: Trained inverse model
        """
        models_dir = os.path.join(self.result_dir, "models")
        
        # Save model state dictionaries
        torch.save(forward_model.state_dict(), os.path.join(models_dir, "forward_model.pth"))
        torch.save(inverse_model.state_dict(), os.path.join(models_dir, "inverse_model.pth"))
        
        # Save complete models (including architecture)
        torch.save(forward_model, os.path.join(models_dir, "forward_model_complete.pth"))
        torch.save(inverse_model, os.path.join(models_dir, "inverse_model_complete.pth"))
        
        self.logger.info(f"Model checkpoints saved to: {models_dir}")
    
    def save_data_artifacts(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        scaler: ZScoreNormalizer,
        material_names: List[str],
        property_names: List[str]
    ):
        """
        Save data artifacts and preprocessing objects.
        
        Preserves all data splits, preprocessing parameters, and metadata
        needed to reproduce the experiment or apply models to new data.
        
        Args:
            X_train, Y_train: Training data
            X_test, Y_test: Test data
            scaler: Fitted normalizer object
            material_names: List of material names
            property_names: List of property names
        """
        data_dir = os.path.join(self.result_dir, "data")
        
        # Save datasets
        np.save(os.path.join(data_dir, "X_train.npy"), X_train)
        np.save(os.path.join(data_dir, "Y_train.npy"), Y_train)
        np.save(os.path.join(data_dir, "X_test.npy"), X_test)
        np.save(os.path.join(data_dir, "Y_test.npy"), Y_test)
        
        # Save normalizer parameters
        scaler_data = {
            "mean": scaler.mean.tolist() if scaler.mean is not None else None,
            "std": scaler.std.tolist() if scaler.std is not None else None
        }
        with open(os.path.join(data_dir, "scaler.json"), 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        # Save names and metadata
        names_data = {
            "material_names": material_names,
            "property_names": property_names
        }
        with open(os.path.join(data_dir, "names.json"), 'w') as f:
            json.dump(names_data, f, indent=2)
        
        self.logger.info(f"Data artifacts saved to: {data_dir}")
    
    def save_predictions_and_results(
        self,
        Y_pred_train: np.ndarray,
        Y_pred_test: np.ndarray,
        candidates: List[np.ndarray],
        predicted_properties: List[np.ndarray],
        target_properties: np.ndarray,
        training_losses: Dict[str, List[float]]
    ):
        """
        Save model predictions and inverse design results.
        
        Args:
            Y_pred_train: Training predictions
            Y_pred_test: Test predictions
            candidates: Inverse design candidate compositions
            predicted_properties: Properties of inverse design candidates
            target_properties: Target properties for inverse design
            training_losses: Training loss histories
        """
        predictions_dir = os.path.join(self.result_dir, "predictions")
        
        # Save predictions
        np.save(os.path.join(predictions_dir, "Y_pred_train.npy"), Y_pred_train)
        np.save(os.path.join(predictions_dir, "Y_pred_test.npy"), Y_pred_test)
        
        # Save inverse design results
        inverse_results = {
            "target_properties": target_properties.tolist(),
            "candidates": [cand.tolist() for cand in candidates],
            "predicted_properties": [pred.tolist() for pred in predicted_properties]
        }
        with open(os.path.join(predictions_dir, "inverse_design_results.json"), 'w') as f:
            json.dump(inverse_results, f, indent=2)
        
        # Save training histories
        with open(os.path.join(predictions_dir, "training_losses.json"), 'w') as f:
            json.dump(training_losses, f, indent=2)
        
        self.logger.info(f"Predictions and results saved to: {predictions_dir}")
    
    def generate_visualizations(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        Y_pred_train: np.ndarray,
        Y_pred_test: np.ndarray,
        candidates: List[np.ndarray],
        predicted_properties: List[np.ndarray],
        target_properties: np.ndarray,
        material_names: List[str],
        property_names: List[str],
        training_losses: Dict[str, List[float]]
    ):
        """
        Generate comprehensive visualization report.
        
        Creates all plots and saves them to the plots directory with
        proper organization and naming.
        
        Args:
            X_train, Y_train: Training data
            X_test, Y_test: Test data
            Y_pred_train, Y_pred_test: Model predictions
            candidates: Inverse design candidates
            predicted_properties: Properties of candidates
            target_properties: Target properties
            material_names: Names of materials
            property_names: Names of properties
            training_losses: Training loss histories
        """
        plots_dir = os.path.join(self.result_dir, "plots")
        
        self.logger.info("Generating comprehensive visualization report...")
        
        create_comprehensive_report(
            X_train, Y_train, X_test, Y_test,
            Y_pred_train, Y_pred_test,
            candidates, predicted_properties, target_properties,
            material_names, property_names,
            training_losses,
            save_dir=plots_dir
        )
        
        self.logger.info(f"Visualizations saved to: {plots_dir}")
    
    def log_experiment_summary(
        self,
        performance_metrics: Dict[str, Any],
        candidates: List[np.ndarray],
        predicted_properties: List[np.ndarray],
        target_properties: np.ndarray,
        material_names: List[str],
        property_names: List[str]
    ):
        """
        Log comprehensive experiment summary.
        
        Args:
            performance_metrics: Model performance metrics
            candidates: Inverse design candidates
            predicted_properties: Properties of candidates
            target_properties: Target properties
            material_names: Names of materials
            property_names: Names of properties
        """
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment ID: {self.unique_id}")
        self.logger.info(f"Results directory: {self.result_dir}")
        
        # Dataset info
        train_samples = len(performance_metrics["train"]["per_property"][property_names[0]])
        test_samples = len(performance_metrics["test"]["per_property"][property_names[0]])
        self.logger.info(f"Dataset: {train_samples + test_samples} samples "
                        f"({train_samples} train, {test_samples} test)")
        self.logger.info(f"Materials: {len(material_names)}")
        self.logger.info(f"Properties: {len(property_names)}")
        
        # Forward model performance
        self.logger.info("Forward Model Performance:")
        for prop_name in property_names:
            train_r2 = performance_metrics["train"]["per_property"][prop_name]["r2"]
            test_r2 = performance_metrics["test"]["per_property"][prop_name]["r2"]
            train_rmse = performance_metrics["train"]["per_property"][prop_name]["rmse"]
            test_rmse = performance_metrics["test"]["per_property"][prop_name]["rmse"]
            self.logger.info(f"  {prop_name}: Train R²={train_r2:.4f} RMSE={train_rmse:.4f}, "
                           f"Test R²={test_r2:.4f} RMSE={test_rmse:.4f}")
        
        # Inverse design results
        self.logger.info("Inverse Design Results:")
        errors = [np.mean((pred - target_properties)**2) for pred in predicted_properties]
        best_idx = np.argmin(errors)
        target_dict = dict(zip(property_names, target_properties))
        self.logger.info(f"  Target: {target_dict}")
        self.logger.info(f"  Generated {len(candidates)} candidates")
        self.logger.info(f"  Best MSE: {errors[best_idx]:.6f}")
        
        # Best composition (only significant components)
        best_comp = candidates[best_idx]
        significant_comp = {mat: float(frac) for mat, frac in zip(material_names, best_comp) 
                          if frac >= 0.05}
        self.logger.info(f"  Best composition: {significant_comp}")
        
        self.logger.info("Files saved:")
        self.logger.info("  - Metadata: experiment_metadata.json")
        self.logger.info("  - Models: models/")
        self.logger.info("  - Data: data/")
        self.logger.info("  - Predictions: predictions/")
        self.logger.info("  - Plots: plots/")
        self.logger.info("  - Log: experiment.log")
        
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 80)
    
    def get_result_dir(self) -> str:
        """Get the full path to the results directory."""
        return self.result_dir
    
    def get_experiment_id(self) -> str:
        """Get the unique experiment identifier."""
        return self.unique_id
