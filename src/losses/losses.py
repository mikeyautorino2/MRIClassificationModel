import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FocalLoss(nn.Module):
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            if self.class_weights.device != inputs.device:
                self.class_weights = self.class_weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C, H, W) or (N, C, D, H, W)
        # targets: (N, H, W) or (N, D, H, W)
        
        num_classes = inputs.shape[1]
        
        # Convert targets to one-hot encoding
        if targets.dim() == inputs.dim() - 1:
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, -1, *range(1, targets.dim())).float()
        else:
            targets_one_hot = targets.float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Flatten spatial dimensions
        inputs_flat = inputs_soft.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Compute Dice coefficient
        intersection = (inputs_flat * targets_flat).sum(dim=-1)
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum(dim=-1) + targets_flat.sum(dim=-1) + self.smooth)
        
        # Dice loss
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class DiceBCELoss(nn.Module):
    def __init__(
        self, 
        dice_weight: float = 0.5, 
        bce_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(inputs, targets)
        
        # For BCE, we need to handle multi-class properly
        if inputs.shape[1] > 1:  # Multi-class
            # Convert to binary masks for each class
            bce = 0
            for c in range(inputs.shape[1]):
                class_targets = (targets == c).float()
                class_inputs = inputs[:, c:c+1]
                bce += self.bce_loss(class_inputs.squeeze(1), class_targets)
            bce /= inputs.shape[1]
        else:  # Binary
            bce = self.bce_loss(inputs.squeeze(1), targets.float())
        
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 0.5, 
        beta: float = 0.5, 
        smooth: float = 1e-6,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.shape[1]
        
        # Convert targets to one-hot encoding
        if targets.dim() == inputs.dim() - 1:
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, -1, *range(1, targets.dim())).float()
        else:
            targets_one_hot = targets.float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Flatten spatial dimensions
        inputs_flat = inputs_soft.view(inputs.shape[0], inputs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Compute Tversky index
        true_pos = (inputs_flat * targets_flat).sum(dim=-1)
        false_neg = (targets_flat * (1 - inputs_flat)).sum(dim=-1)
        false_pos = ((1 - targets_flat) * inputs_flat).sum(dim=-1)
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        
        # Tversky loss
        tversky_loss = 1 - tversky
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


class ComboLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.4,
        focal_weight: float = 0.2,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return -(smooth_targets * log_probs).sum(dim=1).mean()


def create_loss_function(
    loss_name: str,
    num_classes: int = 4,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    
    if loss_name == 'cross_entropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_name == 'weighted_ce':
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif loss_name == 'focal':
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name == 'dice':
        smooth = kwargs.get('smooth', 1e-6)
        return DiceLoss(smooth=smooth)
    
    elif loss_name == 'dice_bce':
        dice_weight = kwargs.get('dice_weight', 0.5)
        bce_weight = kwargs.get('bce_weight', 0.5)
        smooth = kwargs.get('smooth', 1e-6)
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, smooth=smooth)
    
    elif loss_name == 'tversky':
        alpha = kwargs.get('alpha', 0.5)
        beta = kwargs.get('beta', 0.5)
        smooth = kwargs.get('smooth', 1e-6)
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    
    elif loss_name == 'combo':
        ce_weight = kwargs.get('ce_weight', 0.4)
        dice_weight = kwargs.get('dice_weight', 0.4)
        focal_weight = kwargs.get('focal_weight', 0.2)
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        smooth = kwargs.get('smooth', 1e-6)
        return ComboLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            alpha=alpha,
            gamma=gamma,
            smooth=smooth
        )
    
    elif loss_name == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    class_counts = torch.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = len(dataset)
    weights = total_samples / (num_classes * class_counts)
    
    return weights