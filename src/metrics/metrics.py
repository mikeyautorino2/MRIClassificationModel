import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, accuracy_score
)
from scipy.spatial.distance import directed_hausdorff
import torchmetrics
from typing import Dict, List, Optional, Tuple, Union


class MetricsCalculator:
    def __init__(self, num_classes: int = 4, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        if probabilities is None:
            probabilities = F.softmax(predictions, dim=1)
        
        self.all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.all_probabilities.extend(probabilities.cpu().numpy())
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # AUC (macro and per-class)
        if self.num_classes == 2:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            try:
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                
                # Per-class AUC
                for i in range(self.num_classes):
                    y_true_binary = (y_true == i).astype(int)
                    metrics[f'auc_class_{i}'] = roc_auc_score(y_true_binary, y_prob[:, i])
            except ValueError:
                metrics['auc_macro'] = 0.0
                metrics['auc_weighted'] = 0.0
        
        # F1 scores
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
        
        # Precision and Recall
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Sensitivity at 95% specificity
        metrics.update(self._compute_sensitivity_at_specificity(y_true, y_prob))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _compute_sensitivity_at_specificity(self, y_true: np.ndarray, y_prob: np.ndarray, target_specificity: float = 0.95) -> Dict[str, float]:
        metrics = {}
        
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            y_scores = y_prob[:, class_idx]
            
            # Sort by scores
            sorted_indices = np.argsort(y_scores)[::-1]
            y_true_sorted = y_true_binary[sorted_indices]
            
            # Calculate sensitivity and specificity for different thresholds
            n_positive = np.sum(y_true_binary)
            n_negative = len(y_true_binary) - n_positive
            
            if n_positive == 0 or n_negative == 0:
                metrics[f'sensitivity_at_95_specificity_class_{class_idx}'] = 0.0
                continue
            
            tp = 0
            fp = 0
            best_sensitivity = 0.0
            
            for i, label in enumerate(y_true_sorted):
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                
                specificity = (n_negative - fp) / n_negative
                sensitivity = tp / n_positive
                
                if specificity >= target_specificity:
                    best_sensitivity = sensitivity
                else:
                    break
            
            metrics[f'sensitivity_at_95_specificity_class_{class_idx}'] = best_sensitivity
        
        # Overall sensitivity at 95% specificity (macro average)
        sens_values = [v for k, v in metrics.items() if 'sensitivity_at_95_specificity_class_' in k]
        metrics['sensitivity_at_95_specificity_macro'] = np.mean(sens_values) if sens_values else 0.0
        
        return metrics


class SegmentationMetricsCalculator:
    def __init__(self, num_classes: int = 4, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions: (B, C, H, W) or (B, C, D, H, W)
        # targets: (B, H, W) or (B, D, H, W)
        
        pred_masks = predictions.argmax(dim=1).cpu().numpy()
        target_masks = targets.cpu().numpy()
        
        self.all_predictions.extend(pred_masks)
        self.all_targets.extend(target_masks)
    
    def compute_segmentation_metrics(self) -> Dict[str, float]:
        metrics = {}
        
        dice_scores = []
        iou_scores = []
        hausdorff_distances = []
        
        for pred, target in zip(self.all_predictions, self.all_targets):
            # Per-class metrics
            class_dice = []
            class_iou = []
            class_hausdorff = []
            
            for class_idx in range(self.num_classes):
                pred_mask = (pred == class_idx).astype(np.uint8)
                target_mask = (target == class_idx).astype(np.uint8)
                
                # Dice coefficient
                dice = self._compute_dice(pred_mask, target_mask)
                class_dice.append(dice)
                
                # IoU (Jaccard index)
                iou = self._compute_iou(pred_mask, target_mask)
                class_iou.append(iou)
                
                # Hausdorff distance
                if np.sum(pred_mask) > 0 and np.sum(target_mask) > 0:
                    hausdorff = self._compute_hausdorff_distance(pred_mask, target_mask)
                    class_hausdorff.append(hausdorff)
                else:
                    class_hausdorff.append(float('inf'))
            
            dice_scores.append(class_dice)
            iou_scores.append(class_iou)
            hausdorff_distances.append(class_hausdorff)
        
        dice_scores = np.array(dice_scores)
        iou_scores = np.array(iou_scores)
        hausdorff_distances = np.array(hausdorff_distances)
        
        # Overall metrics
        metrics['dice_mean'] = np.mean(dice_scores)
        metrics['dice_std'] = np.std(dice_scores)
        metrics['iou_mean'] = np.mean(iou_scores)
        metrics['iou_std'] = np.std(iou_scores)
        
        # Hausdorff95 (95th percentile)
        finite_hausdorff = hausdorff_distances[np.isfinite(hausdorff_distances)]
        if len(finite_hausdorff) > 0:
            metrics['hausdorff95'] = np.percentile(finite_hausdorff, 95)
            metrics['hausdorff_mean'] = np.mean(finite_hausdorff)
        else:
            metrics['hausdorff95'] = float('inf')
            metrics['hausdorff_mean'] = float('inf')
        
        # Per-class metrics
        for class_idx in range(self.num_classes):
            metrics[f'dice_class_{class_idx}'] = np.mean(dice_scores[:, class_idx])
            metrics[f'iou_class_{class_idx}'] = np.mean(iou_scores[:, class_idx])
            
            finite_hd_class = hausdorff_distances[:, class_idx]
            finite_hd_class = finite_hd_class[np.isfinite(finite_hd_class)]
            if len(finite_hd_class) > 0:
                metrics[f'hausdorff_class_{class_idx}'] = np.mean(finite_hd_class)
            else:
                metrics[f'hausdorff_class_{class_idx}'] = float('inf')
        
        return metrics
    
    def _compute_dice(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        intersection = np.sum(pred * target)
        dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
        return dice
    
    def _compute_iou(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def _compute_hausdorff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        # Get contour points
        pred_points = np.column_stack(np.where(pred > 0))
        target_points = np.column_stack(np.where(target > 0))
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Compute directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)


class UncertaintyMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.entropies = []
        self.predictions = []
        self.targets = []
        self.confidences = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mc_predictions: Optional[List[torch.Tensor]] = None):
        # For MC Dropout uncertainty
        if mc_predictions is not None:
            mc_probs = torch.stack([F.softmax(pred, dim=1) for pred in mc_predictions])
            mean_probs = mc_probs.mean(dim=0)
            
            # Predictive entropy
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            self.entropies.extend(entropy.cpu().numpy())
            
            # Model confidence (max probability)
            confidence = mean_probs.max(dim=1)[0]
            self.confidences.extend(confidence.cpu().numpy())
            
            # Final predictions
            final_predictions = mean_probs.argmax(dim=1)
            self.predictions.extend(final_predictions.cpu().numpy())
        else:
            # Single model prediction
            probs = F.softmax(predictions, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            self.entropies.extend(entropy.cpu().numpy())
            
            confidence = probs.max(dim=1)[0]
            self.confidences.extend(confidence.cpu().numpy())
            
            final_predictions = predictions.argmax(dim=1)
            self.predictions.extend(final_predictions.cpu().numpy())
        
        self.targets.extend(targets.cpu().numpy())
    
    def compute_uncertainty_metrics(self) -> Dict[str, float]:
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        entropies = np.array(self.entropies)
        confidences = np.array(self.confidences)
        
        metrics = {}
        
        # Basic accuracy
        accuracy = np.mean(predictions == targets)
        metrics['accuracy'] = accuracy
        
        # Entropy statistics
        metrics['entropy_mean'] = np.mean(entropies)
        metrics['entropy_std'] = np.std(entropies)
        
        # Confidence statistics
        metrics['confidence_mean'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        
        # Calibration metrics
        correct_predictions = (predictions == targets)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences, correct_predictions)
        metrics['ece'] = ece
        
        # Reliability-confidence correlation
        if len(set(correct_predictions)) > 1:  # Check if there's variance in correctness
            reliability_confidence_corr = np.corrcoef(correct_predictions.astype(float), confidences)[0, 1]
            metrics['reliability_confidence_correlation'] = reliability_confidence_corr if not np.isnan(reliability_confidence_corr) else 0.0
        else:
            metrics['reliability_confidence_correlation'] = 0.0
        
        return metrics
    
    def _compute_ece(self, confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def compute_bootstrap_confidence_interval(
    metrics: Dict[str, float], 
    metric_name: str, 
    n_bootstrap: int = 1000, 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    # This would need the raw data to compute properly
    # For now, return a placeholder
    value = metrics.get(metric_name, 0.0)
    margin = value * 0.05  # 5% margin as placeholder
    alpha = 1 - confidence_level
    return value - margin, value + margin