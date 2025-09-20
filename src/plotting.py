"""
Visualization utilities for materials composition-property modeling results.

This module provides comprehensive plotting functions to visualize training progress,
model performance, composition distributions, and inverse design results for
materials science machine learning workflows.

Functions:
    plot_training_progress: Plot training loss curves over epochs/steps
    plot_property_predictions: Scatter plots comparing predicted vs actual properties
    plot_composition_distributions: Visualize composition distributions and statistics
    plot_inverse_design_results: Visualize inverse design candidates and their properties
    plot_property_correlations: Correlation matrix and pairwise relationships
    plot_composition_ternary: Ternary plots for 3-component systems
    create_comprehensive_report: Generate complete visualization report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


def plot_training_progress(
    forward_losses: List[float],
    inverse_losses: Optional[List[float]] = None,
    inverse_entropies: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training progress for forward and inverse models.
    
    Args:
        forward_losses: List of forward model losses per epoch
        inverse_losses: List of inverse model consistency losses per step
        inverse_entropies: List of entropy values during inverse training
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Forward training progress
    if forward_losses:
        axes[0].plot(forward_losses, 'b-', linewidth=2, label='Forward Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Forward Model Training')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No forward loss data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Forward Model Training')
    
    if inverse_losses is not None and len(inverse_losses) > 0:
        # Inverse consistency loss
        axes[1].plot(inverse_losses, 'r-', linewidth=2, label='Consistency Loss')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Inverse Model Consistency')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No inverse loss data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Inverse Model Consistency')
        
    if inverse_entropies is not None and len(inverse_entropies) > 0:
        # Entropy evolution
        axes[2].plot(inverse_entropies, 'g-', linewidth=2, label='Composition Entropy')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Entropy')
        axes[2].set_title('Composition Diversity (Entropy)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Composition Diversity (Entropy)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_property_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_names: List[str],
    split_name: str = "Test",
    save_path: Optional[str] = None
):
    """
    Create scatter plots comparing predicted vs actual properties.
    
    Args:
        y_true: True property values, shape (N, P)
        y_pred: Predicted property values, shape (N, P)
        property_names: Names of properties
        split_name: Name of the data split (e.g., "Test", "Train")
        save_path: Path to save the plot (optional)
    """
    n_props = len(property_names)
    fig, axes = plt.subplots(1, n_props, figsize=(5*n_props, 4))
    if n_props == 1:
        axes = [axes]
    
    for i, (ax, prop_name) in enumerate(zip(axes, property_names)):
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate R²
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
        
        ax.set_xlabel(f'True {prop_name}')
        ax.set_ylabel(f'Predicted {prop_name}')
        ax.set_title(f'{split_name} Set: {prop_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_composition_distributions(
    compositions: np.ndarray,
    material_names: List[str],
    title: str = "Composition Distributions",
    save_path: Optional[str] = None
):
    """
    Visualize composition distributions across materials.
    
    Args:
        compositions: Composition matrix, shape (N, M)
        material_names: Names of materials
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Box plots of individual material fractions
    axes[0, 0].boxplot([compositions[:, i] for i in range(len(material_names))],
                       labels=[name[:8] for name in material_names])
    axes[0, 0].set_title('Material Fraction Distributions')
    axes[0, 0].set_ylabel('Fraction')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Heatmap of composition correlations
    corr_matrix = np.corrcoef(compositions.T)
    im = axes[0, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Material Fraction Correlations')
    axes[0, 1].set_xticks(range(len(material_names)))
    axes[0, 1].set_yticks(range(len(material_names)))
    axes[0, 1].set_xticklabels([name[:8] for name in material_names], rotation=45)
    axes[0, 1].set_yticklabels([name[:8] for name in material_names])
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Histogram of number of significant components per composition
    n_significant = np.sum(compositions > 0.05, axis=1)  # Components > 5%
    axes[1, 0].hist(n_significant, bins=range(1, len(material_names)+2), alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Number of Significant Components (>5%)')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top materials by average fraction
    avg_fractions = compositions.mean(axis=0)
    sorted_idx = np.argsort(avg_fractions)[::-1]
    top_materials = [material_names[i] for i in sorted_idx[:10]]
    top_fractions = avg_fractions[sorted_idx[:10]]
    
    axes[1, 1].bar(range(len(top_materials)), top_fractions)
    axes[1, 1].set_title('Top 10 Materials by Average Fraction')
    axes[1, 1].set_xlabel('Material')
    axes[1, 1].set_ylabel('Average Fraction')
    axes[1, 1].set_xticks(range(len(top_materials)))
    axes[1, 1].set_xticklabels([name[:8] for name in top_materials], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_inverse_design_results(
    candidates: List[np.ndarray],
    predicted_properties: List[np.ndarray],
    target_properties: np.ndarray,
    material_names: List[str],
    property_names: List[str],
    top_k: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize inverse design results with candidate compositions and properties.
    
    Args:
        candidates: List of candidate compositions
        predicted_properties: List of predicted properties for each candidate
        target_properties: Target property values
        material_names: Names of materials
        property_names: Names of properties
        top_k: Number of top candidates to highlight
        save_path: Path to save the plot (optional)
    """
    if not candidates or not predicted_properties:
        print("No candidates or predicted properties provided for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convert to arrays for easier manipulation
    pred_array = np.array(predicted_properties)
    cand_array = np.array(candidates)
    
    # Calculate errors and get top candidates
    errors = np.mean((pred_array - target_properties) ** 2, axis=1)
    top_idx = np.argsort(errors)[:top_k]
    
    # 1. Property prediction accuracy
    candidate_indices = np.arange(len(candidates))
    
    for i, prop_name in enumerate(property_names):
        axes[0, 0].scatter(candidate_indices, pred_array[:, i], 
                          alpha=0.6, label=f'Predicted {prop_name}', s=30)
        axes[0, 0].axhline(y=target_properties[i], color=f'C{i}', linestyle='--', 
                          linewidth=2, label=f'Target {prop_name}')
    
    # Highlight top candidates - fix the indexing issue
    for idx in top_idx:
        # Plot a star for each property of the top candidates
        for i in range(len(property_names)):
            axes[0, 0].scatter(idx, pred_array[idx, i], marker='*', s=200, 
                              color='red', edgecolor='black', linewidth=1, zorder=5)
    
    axes[0, 0].set_xlabel('Candidate Index')
    axes[0, 0].set_ylabel('Property Value')
    axes[0, 0].set_title('Inverse Design: Property Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Composition diversity (top materials only)
    top_materials_idx = np.argsort(cand_array.mean(axis=0))[-8:]  # Top 8 materials
    
    for i, mat_idx in enumerate(top_materials_idx):
        axes[0, 1].scatter(candidate_indices, cand_array[:, mat_idx], 
                          alpha=0.7, label=material_names[mat_idx][:8], s=30)
    
    axes[0, 1].set_xlabel('Candidate Index')
    axes[0, 1].set_ylabel('Material Fraction')
    axes[0, 1].set_title('Composition Diversity (Top Materials)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    axes[1, 0].hist(errors, bins=20, alpha=0.7, edgecolor='black')
    if len(top_idx) > 0:
        axes[1, 0].axvline(x=errors[top_idx].max(), color='red', linestyle='--', 
                          linewidth=2, label=f'Top-{top_k} threshold')
    axes[1, 0].set_xlabel('Mean Squared Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Property Prediction Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top candidate compositions (stacked bar)
    if len(top_idx) > 0:
        top_compositions = cand_array[top_idx]
        
        bottom = np.zeros(len(top_idx))
        colors = plt.cm.tab20(np.linspace(0, 1, len(material_names)))
        
        for mat_idx in range(len(material_names)):
            fractions = top_compositions[:, mat_idx]
            # Only show materials with significant contribution
            if fractions.max() > 0.05:
                axes[1, 1].bar(range(len(top_idx)), fractions, bottom=bottom, 
                              label=material_names[mat_idx][:8], color=colors[mat_idx])
                bottom += fractions
        
        axes[1, 1].set_xlabel(f'Top-{top_k} Candidates (by MSE)')
        axes[1, 1].set_ylabel('Material Fraction')
        axes[1, 1].set_title('Composition of Best Candidates')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].set_xticks(range(len(top_idx)))
        axes[1, 1].set_xticklabels([f'#{i+1}' for i in range(len(top_idx))])
    else:
        axes[1, 1].text(0.5, 0.5, 'No candidates to display', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Composition of Best Candidates')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_property_correlations(
    properties: np.ndarray,
    property_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix and pairwise relationships between properties.
    
    Args:
        properties: Property matrix, shape (N, P)
        property_names: Names of properties
        save_path: Path to save the plot (optional)
    """
    n_props = len(property_names)
    fig, axes = plt.subplots(n_props, n_props, figsize=(3*n_props, 3*n_props))
    
    if n_props == 1:
        axes = np.array([[axes]])
    elif n_props == 2:
        axes = axes.reshape(2, 2)
    
    for i in range(n_props):
        for j in range(n_props):
            if i == j:
                # Diagonal: histograms
                axes[i, j].hist(properties[:, i], bins=30, alpha=0.7, edgecolor='black')
                axes[i, j].set_title(f'{property_names[i]} Distribution')
            else:
                # Off-diagonal: scatter plots
                axes[i, j].scatter(properties[:, j], properties[:, i], alpha=0.6, s=20)
                
                # Calculate correlation
                corr = np.corrcoef(properties[:, i], properties[:, j])[0, 1]
                axes[i, j].set_title(f'Corr = {corr:.3f}')
            
            if i == n_props - 1:
                axes[i, j].set_xlabel(property_names[j])
            if j == 0:
                axes[i, j].set_ylabel(property_names[i])
            
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_report(
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
    training_losses: Dict[str, List[float]],
    save_dir: Optional[str] = None
):
    """
    Generate a comprehensive visualization report.
    
    Args:
        X_train, Y_train: Training data
        X_test, Y_test: Test data
        Y_pred_train, Y_pred_test: Model predictions
        candidates: Inverse design candidates
        predicted_properties: Properties of inverse design candidates
        target_properties: Target properties for inverse design
        material_names: Names of materials
        property_names: Names of properties
        training_losses: Dictionary with training loss histories
        save_dir: Directory to save plots (optional)
    """
    print("Generating comprehensive visualization report...")
    
    try:
        # 1. Training Progress
        plot_training_progress(
            training_losses.get('forward', []),
            training_losses.get('inverse_consistency', None),
            training_losses.get('inverse_entropy', None),
            save_path=f"{save_dir}/training_progress.png" if save_dir else None
        )
        
        # 2. Property Predictions
        plot_property_predictions(
            Y_train, Y_pred_train, property_names, "Training",
            save_path=f"{save_dir}/train_predictions.png" if save_dir else None
        )
        
        plot_property_predictions(
            Y_test, Y_pred_test, property_names, "Test",
            save_path=f"{save_dir}/test_predictions.png" if save_dir else None
        )
        
        # 3. Composition Analysis
        plot_composition_distributions(
            X_train, material_names, "Training Set Compositions",
            save_path=f"{save_dir}/composition_distributions.png" if save_dir else None
        )
        
        # 4. Property Correlations
        plot_property_correlations(
            Y_train, property_names,
            save_path=f"{save_dir}/property_correlations.png" if save_dir else None
        )
        
        # 5. Inverse Design Results
        if candidates and predicted_properties:
            plot_inverse_design_results(
                candidates, predicted_properties, target_properties,
                material_names, property_names,
                save_path=f"{save_dir}/inverse_design_results.png" if save_dir else None
            )
        else:
            print("Skipping inverse design results plot - no candidates provided")
        
        print("Visualization report complete!")
        
    except Exception as e:
        print(f"Error generating visualization report: {e}")
        import traceback
        traceback.print_exc()
