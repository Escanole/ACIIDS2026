import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from torch.utils.data import DataLoader

from src.data.dataset import CheXpertDataSet
from src.models.architectures import ResNet18, DenseNet121
from src.training.trainer import CheXpertTrainer
from src.utils.gradcam import GradCam

def load_config(config_path):
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

def main():
    # Load configuration
    config = load_config('../config/model_config.yaml')
    
    # Initialize segmentation model
    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    
    # Create transforms
    transform = create_transforms(config)
    
    # Create datasets
    train_dataset = CheXpertDataSet(
        config['paths']['train_csv'],
        '/kaggle/input/chexpert',
        transform=transform,
        policy=config['data']['policy'],
        seg_model=seg_model
    )
    
    val_dataset = CheXpertDataSet(
        config['paths']['valid_csv'],
        '/kaggle/input/chexpert',
        transform=transform,
        seg_model=seg_model
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize models
    model_global = ResNet18(config['model']['num_classes']).cuda()
    model_global = torch.nn.DataParallel(model_global).cuda()
    
    model_local = DenseNet121(config['model']['num_classes']).cuda()
    model_local = torch.nn.DataParallel(model_local).cuda()
    
    # Initialize GradCAM
    grad_cam = GradCam(
        model=model_global.module.resnet18,
        feature_module=model_global.module.resnet18.layer4,
        target_layer_names=["1"],
        use_cuda=True
    )
    
    # Initialize trainer
    trainer = CheXpertTrainer(model_global, model_local, config)
    
    # Train model
    f1_history, auc_history = trainer.train(train_loader, val_loader, seg_model, grad_cam)
    
    print("Training completed!")
    print(f"Final F1: {f1_history[-1]:.4f}")
    print(f"Final AUC: {auc_history[-1]:.4f}")

if __name__ == "__main__":
    main()
