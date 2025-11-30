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
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import CheXpertDataSet
from src.models.architectures import DANN_DenseNet121
from src.training.trainer import CheXpertTrainer
from src.utils.visualization import plot_dann_losses, plot_metrics, plot_gradient_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train DANN Causal Chest X-Ray Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to config file (default: config/model_config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--test', action='store_true',
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
        transforms.Resize(config['data']['resize']),
        transforms.CenterCrop(config['data']['image_size']),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform_list)

seed = 3 #np.random.randint(0, 10000)
print("Currently using seed: ", seed)
def set_seed(seed: int = seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

# call it
set_seed(seed)

# worker seed helper and generator for DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def setup_data_loaders(config, transform):
    print("Creating datasets...")
    
    train_dataset = CheXpertDataSet(
        image_list_file=config['paths']['train_csv'],
        base_path=config['paths']['train_base'],
        transform=transform,
        policy=config['data']['policy'], a1=0.55, b1=0.85
    )
    
    val_dataset = CheXpertDataSet(
        image_list_file=config['paths']['valid_csv'],
        base_path=config['paths']['valid_base'],
        transform=transform
    )

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
        worker_init_fn=seed_worker, generator=g,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4, 
        worker_init_fn=seed_worker, generator=g,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

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
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
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
    
    if args.test:
        print(f"🧪 Testing model only...")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when using --test-only")
        
        trainer.test(test_loader, config['model']['num_classes'], 
                    args.checkpoint, config['class_names'])
    else:
        print(f"Training started at {timestamp}")
    
        try:
            total_losses, task_losses, domain_losses, val_losses, f1s, aucs, aucs_dev, aucs_nodev, all_grad_conflicts = trainer.train(
                train_loader, 
                val_loader,
                config['training']['max_epochs'],
                timestamp,
                checkpoint=args.checkpoint
            )
            
            print("\n" + "="*60)
            print("🎉 Training completed successfully!")
            print("=== DEBUGGING LOSS ARRAYS ===")
            print(f"total_losses: {total_losses}")
            print(f"task_losses: {task_losses}")
            print(f"domain_losses: {domain_losses}")
            print(f"val_losses: {val_losses}")
            print(f"aucs_all: {aucs}")
            print(f"aucs_dev: {aucs_dev}")
            print(f"aucs_nodev: {aucs_nodev}")
            print(f"all_grad_conflicts: {all_grad_conflicts}")

            print(f"\nArray lengths:")
            print(f"total_losses length: {len(total_losses)}")
            print(f"task_losses length: {len(task_losses)}")
            print(f"domain_losses length: {len(domain_losses)}")
            print(f"val_losses length: {len(val_losses)}")

            print(f"\nArray types:")
            print(f"total_losses type: {type(total_losses)}, element type: {type(total_losses[0]) if total_losses else 'empty'}")
            print(f"task_losses type: {type(task_losses)}, element type: {type(task_losses[0]) if task_losses else 'empty'}")

            print(f"\nValue ranges:")
            print(f"total_losses range: {min(total_losses) if total_losses else 'empty'} to {max(total_losses) if total_losses else 'empty'}")
            print(f"task_losses range: {min(task_losses) if task_losses else 'empty'} to {max(task_losses) if task_losses else 'empty'}")
            print(f"domain_losses range: {min(domain_losses) if domain_losses else 'empty'} to {max(domain_losses) if domain_losses else 'empty'}")
            print(f"val_losses range: {min(val_losses) if val_losses else 'empty'} to {max(val_losses) if val_losses else 'empty'}")

            # Check for any weird values
            for i, (total, task, domain, val) in enumerate(zip(total_losses, task_losses, domain_losses, val_losses)):
                if abs(total) > 10 or abs(task) > 10 or abs(domain) > 10 or abs(val) > 10:
                    print(f"⚠️  Epoch {i+1} has extreme values: total={total}, task={task}, domain={domain}, val={val}")
                if np.isnan(total) or np.isnan(task) or np.isnan(domain) or np.isnan(val):
                    print(f"⚠️  Epoch {i+1} has NaN values: total={total}, task={task}, domain={domain}, val={val}")
                if np.isinf(total) or np.isinf(task) or np.isinf(domain) or np.isinf(val):
                    print(f"⚠️  Epoch {i+1} has Inf values: total={total}, task={task}, domain={domain}, val={val}")
            print("="*60)
            
            # Plot results
            plot_dann_losses(total_losses, task_losses, domain_losses, val_losses, per_epoch=True)
            plot_metrics(f1s, aucs, name="ALL")
            plot_metrics(f1s, aucs_dev, name="DEVICE")
            plot_metrics(f1s, aucs_nodev, name="NO-DEVICE")
            plot_gradient_metrics(all_grad_conflicts)
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    main()