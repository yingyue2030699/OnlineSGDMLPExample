# OnlineSGDMLPExample
Example for material composition ~ property predicting using online SGD MLP (pytorch)

# Materials Composition-Property Modeling with Online SGD

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning framework for materials science that implements both forward modeling (composition → properties) and inverse design (properties → compositions) using neural networks trained with online stochastic gradient descent.

## Overview

This project demonstrates a complete pipeline for materials machine learning, featuring:

- **Forward Modeling**: Predict material properties from compositions using multi-layer perceptrons
- **Inverse Design**: Generate material compositions that achieve target properties
- **Online SGD Training**: Efficient training suitable for streaming data scenarios
- **Comprehensive Visualization**: Rich plotting and analysis tools
- **Hyperparameter Studies**: Systematic optimization and performance analysis
- **Reproducible Experiments**: Automated result logging with unique identifiers

## Architecture

### Forward Model
- **Input**: Material compositions (probability simplex: non-negative, sum to 1)
- **Output**: Material properties (multiple continuous values)
- **Architecture**: Multi-layer perceptron with ReLU activations
- **Training**: Online SGD with momentum and weight decay

### Inverse Model
- **Input**: Target properties + noise vector (for diversity)
- **Output**: Material compositions (constrained to probability simplex via softmax)
- **Training**: Forward-consistency loss + entropy regularization
- **Architecture**: Generative MLP with noise injection

## Project Structure

```
materials-property-modeling/
├── src/                                    # Source code
│   ├── utils.py                           # Utilities and normalization
│   ├── models.py                          # Neural network architectures
│   ├── synthetic_data_generator.py        # Dataset generation
│   ├── training.py                        # Training utilities
│   ├── plotting.py                        # Visualization functions
│   ├── output.py                          # Results management
│   ├── online_sgd_mlp_example.py         # Main demonstration script
│   └── hyperparameter_study.py           # Hyperparameter optimization
├── test/                                   # Unit tests
│   ├── unit/                              # Unit test modules
│   │   ├── test_utils.py
│   │   ├── test_models.py
│   │   ├── test_synthetic_data_generator.py
│   │   ├── test_training.py
│   │   ├── test_plotting.py
│   │   └── test_output.py
│   ├── run_tests.py                       # Test runner
│   └── simple_test_runner.py              # Debug test runner
├── results/                               # Experiment results (auto-generated)
├── requirements.txt                       # Python dependencies
├── environment.yml                        # Conda environment
└── README.md                             # This file
```

## Quick Start

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/materials-property-modeling.git
cd materials-property-modeling

# Create and activate conda environment
conda env create -f environment.yml
conda activate materials-modeling
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/your-username/materials-property-modeling.git
cd materials-property-modeling

# Create virtual environment
python -m venv materials_env
source materials_env/bin/activate  # On Windows: materials_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Run the main demonstration:
```bash
cd src
python online_sgd_mlp_example.py
```

This will:
1. Generate synthetic materials dataset
2. Train forward and inverse models
3. Demonstrate inverse design
4. Create comprehensive visualizations
5. Save all results to `results/YYYYMMDD_HHMMSS_hash/`

#### Run hyperparameter study:
```bash
cd src
python hyperparameter_study.py
```

#### Run tests:
```bash
cd test
python run_tests.py
```

## Features

### Synthetic Data Generation
- **Realistic Compositions**: Dirichlet distribution ensures valid probability simplexes
- **Nonlinear Properties**: Complex property-composition relationships with interactions
- **Multiple Scales**: Properties with different ranges (0-100, -3 to +3, 0-1)
- **Configurable Noise**: Adjustable noise levels for realistic scenarios

### Neural Network Models
- **ForwardNet**: Composition → Properties mapping
- **InverseNet**: Properties → Compositions with simplex constraints
- **Flexible Architecture**: Configurable hidden layers and dimensions
- **Gradient Flow**: Proper gradient computation for end-to-end training

### Training Features
- **Online SGD**: Mini-batch streaming with data shuffling
- **Consistency Training**: Inverse model trained against frozen forward model
- **Entropy Regularization**: Encourages diverse composition generation
- **Loss Tracking**: Comprehensive monitoring of training progress

### Visualization Suite
- **Training Progress**: Loss curves and convergence analysis
- **Model Performance**: Prediction accuracy with R² and RMSE
- **Composition Analysis**: Distribution plots and correlation matrices
- **Inverse Design Results**: Candidate analysis and property matching
- **Hyperparameter Studies**: Performance comparison across configurations

### Experiment Management
- **Unique Identifiers**: Timestamp + configuration hash for each experiment
- **Comprehensive Logging**: Detailed logs with configurable levels
- **Metadata Tracking**: Complete experiment configuration and system info
- **Result Preservation**: Models, data, predictions, and visualizations saved
- **Reproducible Results**: Seed management for consistent outcomes

## Detailed Usage

### Configuration

The main script uses a comprehensive configuration system:

```python
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
            "weight_decay": 1e-4
        },
        "inverse": {
            "lr": 0.01,
            "steps": 4000,
            "batch_size": 64,
            "entropy_weight": 8e-3,
            "l2_logits_weight": 1e-4
        }
    },
    "inverse_design": {
        "target_properties": [80.0, 2.0, 0.7],
        "n_candidates": 50
    }
}
```

### Custom Datasets

To use your own materials data:

```python
from utils import ZScoreNormalizer
from models import ForwardNet, InverseNet
from training import train_forward_online, train_inverse_online

# Load your data
X = your_compositions  # Shape: (N, M) - compositions sum to 1
Y = your_properties   # Shape: (N, P) - property values

# Normalize properties
scaler = ZScoreNormalizer()
scaler.fit(Y_train)
Y_train_norm = scaler.transform(Y_train)

# Create and train models
forward_model = ForwardNet(in_dim=M, out_dim=P, hidden=128)
train_forward_online(forward_model, X_train, Y_train_norm, 
                    lr=0.01, epochs=10, batch_size=32)

# Use trained models for prediction and inverse design
```

### Hyperparameter Optimization

Customize the hyperparameter study:

```python
# Define parameter grid
param_grid = {
    "training.forward.lr": [0.005, 0.01, 0.02, 0.05],
    "training.forward.epochs": [3, 5, 8, 12],
    "model_architecture.forward.hidden_size": [32, 64, 128, 256],
    "training.inverse.entropy_weight": [1e-3, 5e-3, 1e-2, 2e-2]
}

# Run study
study = HyperparameterStudy(base_config, param_grid, n_runs=3)
study.run_study()
```

## Results and Analysis

### Experiment Output

Each experiment generates a unique directory structure:

```
results/YYYYMMDD_HHMMSS_hash123/
├── experiment.log                  # Detailed execution log
├── experiment_metadata.json        # Complete configuration and system info
├── models/                        # Trained model checkpoints
│   ├── forward_model.pth          # State dict
│   ├── forward_model_complete.pth # Complete model
│   ├── inverse_model.pth          # State dict
│   └── inverse_model_complete.pth # Complete model
├── data/                          # Dataset and preprocessing
│   ├── X_train.npy, Y_train.npy   # Training data
│   ├── X_test.npy, Y_test.npy     # Test data
│   ├── scaler.json                # Normalization parameters
│   └── names.json                 # Material and property names
├── predictions/                   # Model outputs
│   ├── Y_pred_train.npy           # Training predictions
│   ├── Y_pred_test.npy            # Test predictions
│   ├── inverse_design_results.json # Inverse design candidates
│   └── training_losses.json       # Training history
└── plots/                         # Visualizations
    ├── training_progress.png       # Loss curves
    ├── train_predictions.png       # Training performance
    ├── test_predictions.png        # Test performance
    ├── composition_distributions.png # Data analysis
    ├── property_correlations.png   # Property relationships
    └── inverse_design_results.png  # Inverse design analysis
```

### Performance Metrics

The framework tracks comprehensive metrics:

- **Forward Model**: MSE, RMSE, MAE, R² (overall and per-property)
- **Inverse Model**: Consistency loss, entropy, success rates
- **Inverse Design**: Mean/min/std MSE, success rates at different thresholds

### Visualization Examples

1. **Training Progress**: Monitor convergence of both models
2. **Property Predictions**: Scatter plots with perfect prediction lines
3. **Composition Analysis**: Distribution plots and correlation matrices
4. **Inverse Design**: Candidate ranking and property matching
5. **Hyperparameter Studies**: Performance comparison across configurations

#### Memory Issues
```bash
# For large datasets, reduce batch size or dataset size:
config["dataset"]["n_samples"] = 2000  # Reduce from 6000
config["training"]["forward"]["batch_size"] = 16  # Reduce from 32
```

#### Visualization Issues
```bash
# If plots don't display, ensure matplotlib backend:
export MPLBACKEND=Agg  # For headless systems
# Or install GUI backend:
pip install PyQt5  # or tkinter
```

### Inverse Design
- Sanchez-Lengeling, B., & Aspuru-Guzik, A. "Inverse molecular design using machine learning." *Science* 361.6400 (2018): 360-365.
- Kim, B., et al. "Deep-learning-accelerated inverse design of materials." *npj Computational Materials* 7.1 (2021): 1-8.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Open Source Contributors** for various tools and libraries used
