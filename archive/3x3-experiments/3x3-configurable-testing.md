# Configurable 3x3 Testing System

## Overview

The configurable 3x3 testing system allows you to easily run different experiments with various datasets, architectures, and training configurations without modifying code.

## Usage

### 1. Individual Experiments

#### Using Preset Configurations
```bash
# Basic architecture (one-hot features, single hidden layer)
cargo run --release --bin train_3x3_configurable basic

# Spatial features (one-hot + coordinates, two hidden layers)
cargo run --release --bin train_3x3_configurable spatial

# Enhanced features (one-hot + coordinates + gradients, three hidden layers)
cargo run --release --bin train_3x3_configurable enhanced
```

#### Using Custom Configuration Files
```bash
# Run with a specific config file
cargo run --release --bin train_3x3_configurable config configs/enhanced_3x3.json

# Available config files:
# - configs/basic_3x3.json
# - configs/spatial_3x3.json  
# - configs/enhanced_3x3.json
# - configs/deep_3x3.json
```

### 2. Batch Experiments

```bash
# Run all default experiments
cargo run --release --bin run_experiments

# Run specific config files
cargo run --release --bin run_experiments configs/basic_3x3.json configs/spatial_3x3.json

# Run custom set of experiments
cargo run --release --bin run_experiments configs/enhanced_3x3.json configs/deep_3x3.json
```

## Configuration Options

### Feature Types
- **Basic**: One-hot encoding (90 features)
- **Spatial**: One-hot + center-relative coordinates (108 features)
- **Enhanced**: One-hot + coordinates + intensity gradients (126 features)

### Architecture Options
- **Hidden layers**: Array of layer sizes (e.g., `[64]`, `[128, 64]`, `[256, 128, 64]`)
- **Activation**: Currently supports "relu"
- **Dropout**: Optional dropout rate (e.g., `null`, `0.1`, `0.2`)
- **Layer normalization**: Boolean flag for stability

### Training Parameters
- **Learning rate**: Step size for optimization (0.001 - 0.01)
- **Steps**: Maximum training iterations
- **Weight decay**: Regularization strength
- **Patience**: Steps without improvement before early stopping
- **Evaluation frequency**: How often to check accuracy

### Output Options
- **Save directory**: Where to store results
- **Verbose**: Detailed progress logging
- **Plots**: Generate performance plots (future feature)

## Results Structure

Each experiment generates:
```
experiments/[experiment_name]/
├── config.json      # Configuration used
├── results.json      # Training results and metrics
└── [additional files] # Future: plots, checkpoints, etc.
```

Batch results generate:
```
batch_results/
├── batch_summary.json  # Comparison of all experiments
├── [experiment_1]/      # Individual experiment results
├── [experiment_2]/
└── ...
```

## Example Configurations

### Basic 3x3 (Fast Testing)
```json
{
  "name": "basic_3x3_test",
  "dataset": "eyespot_training_3x3.txt",
  "feature_type": "Basic",
  "architecture": {
    "hidden_layers": [64],
    "activation": "relu",
    "dropout": null,
    "layer_norm": false
  },
  "training": {
    "learning_rate": 0.01,
    "steps": 5000,
    "batch_size": 1,
    "weight_decay": 0.01,
    "patience": 1000,
    "eval_frequency": 100
  }
}
```

### Enhanced 3x3 (Best Performance)
```json
{
  "name": "enhanced_3x3_test",
  "dataset": "eyespot_training_3x3.txt",
  "feature_type": "Enhanced",
  "architecture": {
    "hidden_layers": [256, 128, 64],
    "activation": "relu",
    "dropout": 0.1,
    "layer_norm": true
  },
  "training": {
    "learning_rate": 0.003,
    "steps": 50000,
    "batch_size": 1,
    "weight_decay": 0.01,
    "patience": 5000,
    "eval_frequency": 1000
  }
}
```

### Deep 3x3 (Maximum Capacity)
```json
{
  "name": "deep_3x3_test",
  "dataset": "eyespot_training_3x3.txt",
  "feature_type": "Enhanced",
  "architecture": {
    "hidden_layers": [512, 256, 128, 64],
    "activation": "relu",
    "dropout": 0.2,
    "layer_norm": true
  },
  "training": {
    "learning_rate": 0.001,
    "steps": 100000,
    "batch_size": 1,
    "weight_decay": 0.01,
    "patience": 10000,
    "eval_frequency": 2000
  }
}
```

## Performance Benchmarks

Based on current testing:

| Configuration | Best Accuracy | Training Time | Convergence |
|---------------|----------------|---------------|-------------|
| Basic         | ~20%          | ~2s          | Fast         |
| Spatial       | ~48%          | ~6s          | Medium       |
| Enhanced       | ~53%          | ~29s         | Slow         |
| Deep          | TBD           | TBD           | Very Slow    |

## Custom Datasets

To use different datasets:

1. **Create your dataset file** (same format as `eyespot_training_3x3.txt`)
2. **Update the config**:
```json
{
  "dataset": "my_custom_dataset.txt",
  // ... other config
}
```

3. **Run the experiment**:
```bash
cargo run --release --bin train_3x3_configurable config configs/my_experiment.json
```

## Creating New Configurations

1. **Copy an existing config**:
```bash
cp configs/enhanced_3x3.json configs/my_experiment.json
```

2. **Edit the configuration** to match your needs

3. **Test your configuration**:
```bash
cargo run --release --bin train_3x3_configurable config configs/my_experiment.json
```

## Batch Testing Workflow

For systematic testing:

1. **Define your experiment set**:
```bash
# Create multiple config files for different variables
configs/test_lr_001.json
configs/test_lr_005.json  
configs/test_lr_010.json
```

2. **Run batch comparison**:
```bash
cargo run --release --bin run_experiments configs/test_lr_*.json
```

3. **Analyze results**:
```bash
# Check the summary
cat batch_results/batch_summary.json | jq '.comparison'

# Compare individual experiments
ls batch_results/
```

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the dataset file exists and path is correct
2. **Memory issues**: Reduce hidden layer sizes or batch size
3. **Slow convergence**: Increase learning rate or reduce model complexity
4. **Overfitting**: Increase weight decay or add dropout

### Performance Tips

1. **Start simple**: Begin with basic configuration to establish baseline
2. **Gradual complexity**: Increase model size incrementally
3. **Monitor convergence**: Use early stopping to avoid wasted training
4. **Batch testing**: Use batch runner for systematic comparison

## Next Steps

The configurable system makes it easy to:

1. **Test different datasets** quickly
2. **Compare architectures** systematically  
3. **Optimize hyperparameters** efficiently
4. **Reproduce experiments** reliably
5. **Scale to larger experiments** when needed

This tooling provides the foundation for systematic 3x3 vision block development and optimization.
