import os
import sys
import yaml
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import CheXpertDataSet
from src.models.architectures import DANN_DenseNet121
from src.training.trainer import CheXpertTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train DANN Causal Chest X-Ray Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to config file (default: config/model_config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing, skip training')
    return parser.parse_args()

def load_config(config_path='config/model_config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_transforms(config):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_list = [
        transforms.RandomResizedCrop(config['data']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform_list)

def setup_data_loaders(config, transform):
    print("Creating datasets...")
    
    train_dataset = CheXpertDataSet(
        image_list_file=config['paths']['train_csv'],
        base_path=config['paths']['train_base'],
        transform=transform,
        policy=config['data']['policy']
    )
    
    val_dataset = CheXpertDataSet(
        image_list_file=config['paths']['valid_csv'],
        base_path=config['paths']['valid_base'],
        transform=transform
    )
    
    # For testing (optional)
    test_dataset = CheXpertDataSet(
        image_list_file=config['paths']['test_csv'],
        base_path=config['paths']['test_base'],
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def plot_results(total_losses, task_losses, domain_losses, val_losses, f1s, aucs):
    """Plot training results"""
    epochs = range(1, len(total_losses) + 1)
    
    # Plot losses
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss components
    ax1.plot(epochs, total_losses, 'r-', label='Total Loss')
    ax1.plot(epochs, task_losses, 'b-', label='Task Loss')
    ax1.plot(epochs, domain_losses, 'g-', label='Domain Loss')
    ax1.plot(epochs, val_losses, 'orange', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True)
    
    # Task vs Val loss
    ax2.plot(epochs, task_losses, 'b-', label='Train Task Loss')
    ax2.plot(epochs, val_losses, 'orange', label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Task Loss vs Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    # F1 Score
    ax3.plot(epochs, f1s, 'g-', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Evolution')
    ax3.grid(True)
    
    # AUROC
    if isinstance(aucs[0], np.ndarray):
        mean_aucs = [np.nanmean(auc) for auc in aucs]
        ax4.plot(epochs, mean_aucs, 'purple', marker='s')
    else:
        ax4.plot(epochs, aucs, 'purple', marker='s')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUROC')
    ax4.set_title('AUROC Evolution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    args = parse_args()
    
    print("🚀 Starting DANN Causal Chest X-Ray Training...")
    print(f"📄 Using config file: {args.config}")
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    print(f"Model: {config['model']['architecture']}")
    print(f"Train CSV: {config['paths']['train_csv']}")
    print(f"Valid CSV: {config['paths']['valid_csv']}")
    print(f"Test CSV: {config['paths']['test_csv']}")
    
    # Create directories
    os.makedirs('models/checkpoints/best_loss', exist_ok=True)
    os.makedirs('models/checkpoints/best_auc', exist_ok=True)
    os.makedirs('models/checkpoints/last', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Setup data
    transform = create_transforms(config)
    train_loader, val_loader, test_loader = setup_data_loaders(config, transform)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Setup model
    print("Initializing DANN model...")
    model = DANN_DenseNet121(
        task_out_size=config['model']['num_classes'],
        domain_out_size=config['model']['domain_classes'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    
    # Setup trainer
    trainer = CheXpertTrainer(model, config)
    
    # Start training or testing
    timestamp = time.strftime("%d%m%Y-%H%M%S")
    
    if args.test_only:
        print(f"🧪 Testing model only...")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when using --test-only")
        
        trainer.test(test_loader, config['model']['num_classes'], 
                    args.checkpoint, config['class_names'])
    else:
        print(f"Training started at {timestamp}")
        
        try:
            total_losses, task_losses, domain_losses, val_losses, f1s, aucs = trainer.train(
                train_loader, 
                val_loader,
                config['training']['max_epochs'],
                timestamp,
                checkpoint=args.checkpoint
            )
            
            print("\n" + "="*60)
            print("🎉 Training completed successfully!")
            print(f"Final Task Loss: {task_losses[-1]:.4f}")
            print(f"Final Val Loss: {val_losses[-1]:.4f}")
            print(f"Final F1 Score: {f1s[-1]:.4f}")
            
            if isinstance(aucs[-1], np.ndarray):
                final_auc = np.nanmean(aucs[-1])
                print(f"Final Mean AUROC: {final_auc:.4f}")
            else:
                print(f"Final AUROC: {aucs[-1]:.4f}")
            print("="*60)
            
            # Plot results
            plot_results(total_losses, task_losses, domain_losses, val_losses, f1s, aucs)
            
            # Optional: Test the model
            print("\n🧪 Testing model...")
            trainer.test(test_loader, config['model']['num_classes'], 
                        f"models/checkpoints/best_auc/best_auc_epoch{len(aucs)-1}_{timestamp}.pth.tar", 
                        config['class_names'])
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    main()
