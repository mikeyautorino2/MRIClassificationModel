#!/usr/bin/env python3
"""
Main training script for MRI classification and segmentation.

Usage:
    python train.py --config configs/base_config.yaml
    python train.py --config configs/efficientnet_focal.yaml --wandb
    python train.py --config configs/resnet_3d.yaml --gpu 0
"""

import argparse
import yaml
import torch
import wandb
import random
import numpy as np
from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.classification_2d import create_model
from src.models.classification_3d import create_3d_model
from src.models.segmentation import create_segmentation_model
from src.utils.training import setup_training


def parse_args():
    parser = argparse.ArgumentParser(description='Train MRI classification/segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_from_config(config: dict):
    """Create model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] in ['efficientnet_b0', 'resnet50', 'slice_efficientnet_b0', 'slice_resnet50']:
        return create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config.get('pretrained', True),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            use_mc_dropout=model_config.get('use_mc_dropout', False),
            aggregation_method=model_config.get('aggregation_method', 'mean')
        )
    elif model_config['name'] in ['r3d_18', 'videoresnet3d', 'efficientnet3d', 'densenet3d']:
        return create_3d_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config.get('pretrained', True),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            use_mc_dropout=model_config.get('use_mc_dropout', False)
        )
    elif model_config['name'] in ['unet2d', 'unet3d', 'attention_unet', 'resunet']:
        return create_segmentation_model(
            model_name=model_config['name'],
            n_channels=model_config.get('n_channels', 3),
            n_classes=model_config['n_classes'],
            bilinear=model_config.get('bilinear', False)
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override wandb setting if specified
    if args.wandb:
        config['training']['use_wandb'] = True
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if config['training'].get('use_wandb', False):
        wandb.init(
            project="mri-classification",
            name=config['experiment']['name'],
            tags=config['experiment']['tags'],
            notes=config['experiment']['notes'],
            config=config
        )
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        use_weighted_sampling=config['data'].get('use_weighted_sampling', True)
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Debug mode - use smaller dataset
    if args.debug:
        print("Debug mode: Using smaller dataset")
        from torch.utils.data import Subset
        train_indices = list(range(min(100, len(train_loader.dataset))))
        val_indices = list(range(min(50, len(val_loader.dataset))))
        train_loader.dataset = Subset(train_loader.dataset, train_indices)
        val_loader.dataset = Subset(val_loader.dataset, val_indices)
    
    # Create model
    print(f"Creating model: {config['model']['name']}")
    model = create_model_from_config(config)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    print("Setting up training...")
    trainer, early_stopping = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Start training
    print("Starting training...")
    try:
        training_history = trainer.train(
            num_epochs=config['training']['num_epochs'],
            early_stopping=early_stopping,
            save_best=True,
            use_mixup=config['training'].get('use_mixup', True),
            use_cutmix=config['training'].get('use_cutmix', True),
            use_uncertainty=config['evaluation'].get('use_uncertainty', False),
            log_interval=config['training'].get('log_interval', 1)
        )
        
        print("Training completed successfully!")
        
        # Save final results
        results_path = Path(config['training']['save_dir']) / 'training_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(training_history, f)
        print(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        if config['training'].get('use_wandb', False):
            wandb.finish()


if __name__ == '__main__':
    main()