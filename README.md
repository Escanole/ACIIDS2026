# Causal Chest X-Ray Analysis

A DANN (Domain Adversarial Neural Network) implementation for causal analysis of chest X-ray images using the CheXpert dataset.

## Features
- Domain adversarial training to handle support device bias
- Attention mechanisms for better feature learning
- Configurable training parameters via YAML config
- Support for local dataset paths
- Command line config selection

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update the config file `config/model_config.yaml` with your dataset paths

## Usage

### Basic Training
```bash
python train.py
```

### Training with Custom Config
```bash
python train.py --config config/model_config_local.yaml
```

### Resume Training from Checkpoint
```bash
python train.py --config config/model_config.yaml --checkpoint models/checkpoints/last/last_epoch5_12345678-123456.pth.tar
```

### Test Only (Skip Training)
```bash
python train.py --config config/model_config.yaml --test-only --checkpoint models/checkpoints/best_auc/best_auc_epoch10_12345678-123456.pth.tar
```

## Command Line Arguments

- `--config`: Path to YAML config file (default: `config/model_config.yaml`)
- `--checkpoint`: Path to checkpoint file to resume training from
- `--test-only`: Skip training and only run testing (requires `--checkpoint`)

## Configuration

Create different config files for different experiments:
- `config/model_config.yaml` - Main configuration
- `config/model_config_local.yaml` - Local testing with smaller dataset
- `config/model_config_production.yaml` - Production settings

## Output

- Model checkpoints saved to `models/checkpoints/`
- Training plots saved to `results/`
- Logs printed to console

## Examples

```bash
# Quick local test with small dataset
python train.py --config config/model_config_local.yaml

# Full training with production settings
python train.py --config config/model_config_production.yaml

# Test a trained model
python train.py --config config/model_config.yaml --test-only --checkpoint models/checkpoints/best_auc/best_model.pth.tar
```
