import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.cuda.amp import GradScaler, autocast
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import time
from tqdm import tqdm
import logging
from collections import defaultdict
import math

from ..losses.losses import create_loss_function
from ..metrics.metrics import MetricsCalculator, SegmentationMetricsCalculator, UncertaintyMetrics


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, eta_min: float = 0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))


class MixUp:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random box
        H, W = x.shape[-2:]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cpu',
        use_amp: bool = True,
        use_wandb: bool = False,
        save_dir: str = 'checkpoints',
        task_type: str = 'classification'  # 'classification' or 'segmentation'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.task_type = task_type
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        if use_amp:
            self.scaler = GradScaler()
        
        # Initialize metrics calculators
        if task_type == 'classification':
            self.train_metrics = MetricsCalculator(device=device)
            self.val_metrics = MetricsCalculator(device=device)
        else:
            self.train_metrics = SegmentationMetricsCalculator(device=device)
            self.val_metrics = SegmentationMetricsCalculator(device=device)
        
        self.uncertainty_metrics = UncertaintyMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        
        # Augmentation techniques
        self.mixup = MixUp(alpha=1.0)
        self.cutmix = CutMix(alpha=1.0)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, use_mixup: bool = True, use_cutmix: bool = True) -> Dict[str, float]:
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply augmentation techniques
            if use_mixup and np.random.random() < 0.5:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
                mixed_target = True
            elif use_cutmix and np.random.random() < 0.5:
                inputs, targets_a, targets_b, lam = self.cutmix(inputs, targets)
                mixed_target = True
            else:
                mixed_target = False
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take logits if attention weights are returned
                    
                    if mixed_target:
                        loss = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)
                    else:
                        loss = self.loss_fn(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if mixed_target:
                    loss = lam * self.loss_fn(outputs, targets_a) + (1 - lam) * self.loss_fn(outputs, targets_b)
                else:
                    loss = self.loss_fn(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
            
            # Update metrics (use original targets for mixed augmentation)
            if not mixed_target:
                if self.task_type == 'classification':
                    self.train_metrics.update(outputs, targets)
                else:
                    self.train_metrics.update(outputs, targets)
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        # Compute epoch metrics
        avg_loss = running_loss / num_batches
        
        if self.task_type == 'classification':
            metrics = self.train_metrics.compute_classification_metrics()
        else:
            metrics = self.train_metrics.compute_segmentation_metrics()
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, use_uncertainty: bool = False) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()
        if use_uncertainty:
            self.uncertainty_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.loss_fn(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.loss_fn(outputs, targets)
                
                # Update metrics
                if self.task_type == 'classification':
                    self.val_metrics.update(outputs, targets)
                    if use_uncertainty:
                        self.uncertainty_metrics.update(outputs, targets)
                else:
                    self.val_metrics.update(outputs, targets)
                
                running_loss += loss.item()
        
        # Compute epoch metrics
        avg_loss = running_loss / num_batches
        
        if self.task_type == 'classification':
            metrics = self.val_metrics.compute_classification_metrics()
            if use_uncertainty:
                uncertainty_metrics = self.uncertainty_metrics.compute_uncertainty_metrics()
                metrics.update(uncertainty_metrics)
        else:
            metrics = self.val_metrics.compute_segmentation_metrics()
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        save_best: bool = True,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        use_uncertainty: bool = False,
        log_interval: int = 1
    ) -> Dict[str, List[float]]:
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(use_mixup=use_mixup, use_cutmix=use_cutmix)
            
            # Validation
            val_metrics = self.validate_epoch(use_uncertainty=use_uncertainty)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, CosineWarmupScheduler):
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()
            
            # Update training history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            epoch_time = time.time() - start_time
            
            # Logging
            if epoch % log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                if self.task_type == 'classification':
                    self.logger.info(
                        f"Train Acc: {train_metrics.get('accuracy', 0):.4f}, "
                        f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                        f"Val AUC: {val_metrics.get('auc_macro', 0):.4f}"
                    )
                else:
                    self.logger.info(
                        f"Train Dice: {train_metrics.get('dice_mean', 0):.4f}, "
                        f"Val Dice: {val_metrics.get('dice_mean', 0):.4f}"
                    )
            
            # Wandb logging
            if self.use_wandb:
                log_dict = {}
                for key, value in train_metrics.items():
                    log_dict[f'train/{key}'] = value
                for key, value in val_metrics.items():
                    log_dict[f'val/{key}'] = value
                log_dict['epoch'] = epoch
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict)
            
            # Save best model
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_metrics['loss'], self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        self.logger.info("Training completed")
        return dict(self.training_history)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_history': dict(self.training_history)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    **kwargs
) -> Optional[Any]:
    
    if scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_name.lower() == 'cosine_warmup':
        return CosineWarmupScheduler(optimizer, warmup_epochs, num_epochs, **kwargs)
    elif scheduler_name.lower() == 'cosine_restart':
        T_0 = kwargs.get('T_0', 10)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, **kwargs)
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def setup_training(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> Trainer:
    
    # Create loss function
    loss_fn = create_loss_function(
        loss_name=config['loss']['name'],
        num_classes=config['model']['num_classes'],
        **config['loss'].get('params', {})
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config['optimizer']['name'],
        learning_rate=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 1e-4),
        **config['optimizer'].get('params', {})
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=config['scheduler']['name'],
        num_epochs=config['training']['num_epochs'],
        **config['scheduler'].get('params', {})
    )
    
    # Create early stopping
    early_stopping = None
    if config['training'].get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping'].get('patience', 7),
            min_delta=config['training']['early_stopping'].get('min_delta', 0.0)
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=config['training'].get('use_amp', True),
        use_wandb=config['training'].get('use_wandb', False),
        save_dir=config['training'].get('save_dir', 'checkpoints'),
        task_type=config['training'].get('task_type', 'classification')
    )
    
    return trainer, early_stopping