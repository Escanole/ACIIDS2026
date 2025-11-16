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

### Command Line Arguments

- `--config`: Path to YAML config file (default: `config/model_config.yaml`)
- `--checkpoint`: Path to checkpoint file to resume training from
- 
### Basic Training
```bash
python train.py
```

### Resume Training from Checkpoint
```bash
python train.py --config config/model_config.yaml --checkpoint models/checkpoints/last/last_epoch5_12345678-123456.pth.tar
```

## Output

- Model checkpoints saved to `models/checkpoints/`
- Training plots saved to `results/`
- Logs printed to console
