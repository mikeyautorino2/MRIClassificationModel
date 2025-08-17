import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import copy


class MCDropoutPredictor:
    def __init__(self, model: nn.Module, n_samples: int = 100, device: str = 'cpu'):
        self.model = model
        self.n_samples = n_samples
        self.device = device
        self.model.to(device)
    
    def enable_dropout(self, model: nn.Module):
        """Enable dropout layers during inference for MC Dropout."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
            # Also enable custom MCDropout layers
            elif hasattr(module, '__class__') and 'MCDropout' in module.__class__.__name__:
                module.train()
    
    def predict_with_uncertainty(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()
        self.enable_dropout(self.model)
        
        all_predictions = []
        all_targets = []
        all_mc_samples = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                mc_predictions = []
                
                # Generate MC samples
                for _ in range(self.n_samples):
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take logits if attention weights are returned
                    mc_predictions.append(outputs.cpu())
                
                mc_predictions = torch.stack(mc_predictions)  # (n_samples, batch_size, n_classes)
                
                # Compute statistics
                mc_probs = F.softmax(mc_predictions, dim=-1)
                mean_probs = mc_probs.mean(dim=0)
                predictions = mean_probs.argmax(dim=-1)
                
                all_predictions.extend(predictions.numpy())
                all_targets.extend(targets.cpu().numpy())
                all_mc_samples.append(mc_probs.numpy())
        
        # Combine all MC samples
        all_mc_samples = np.concatenate(all_mc_samples, axis=1)  # (n_samples, total_samples, n_classes)
        
        # Compute uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(all_mc_samples)
        
        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'mc_samples': all_mc_samples,
            'uncertainty_metrics': uncertainty_metrics
        }
    
    def _compute_uncertainty_metrics(self, mc_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various uncertainty metrics from MC samples."""
        # mc_samples shape: (n_samples, total_samples, n_classes)
        
        mean_probs = mc_samples.mean(axis=0)  # (total_samples, n_classes)
        
        # Predictive entropy (aleatoric uncertainty)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        
        # Mutual information (epistemic uncertainty)
        entropies_per_sample = -np.sum(mc_samples * np.log(mc_samples + 1e-8), axis=2)  # (n_samples, total_samples)
        expected_entropy = entropies_per_sample.mean(axis=0)  # (total_samples,)
        mutual_information = predictive_entropy - expected_entropy
        
        # Variance of predictions
        prediction_variance = mc_samples.var(axis=0).max(axis=1)  # (total_samples,)
        
        # Standard deviation of max probability
        max_probs = mc_samples.max(axis=2)  # (n_samples, total_samples)
        max_prob_std = max_probs.std(axis=0)  # (total_samples,)
        
        return {
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_information,
            'prediction_variance': prediction_variance,
            'max_prob_std': max_prob_std,
            'mean_probs': mean_probs
        }


class DeepEnsemble:
    def __init__(self, model_configs: List[Dict], device: str = 'cpu'):
        self.models = []
        self.device = device
        self.model_configs = model_configs
        
        # Initialize ensemble models
        for config in model_configs:
            model = self._create_model_from_config(config)
            model.to(device)
            self.models.append(model)
    
    def _create_model_from_config(self, config: Dict) -> nn.Module:
        """Create a model from configuration dictionary."""
        # This would need to be implemented based on your model factory
        # For now, return a placeholder
        from ..models.classification_2d import create_model
        return create_model(**config)
    
    def load_ensemble(self, checkpoint_paths: List[str]):
        """Load pre-trained ensemble models."""
        for model, checkpoint_path in zip(self.models, checkpoint_paths):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
    
    def predict_with_uncertainty(self, dataloader) -> Dict[str, np.ndarray]:
        all_predictions = []
        all_targets = []
        ensemble_predictions = []
        
        # Get predictions from each model in the ensemble
        for model in self.models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    probs = F.softmax(outputs, dim=1)
                    model_predictions.append(probs.cpu().numpy())
                    
                    if len(all_targets) == 0:  # Only collect targets once
                        all_targets.extend(targets.numpy())
            
            ensemble_predictions.append(np.concatenate(model_predictions, axis=0))
        
        ensemble_predictions = np.stack(ensemble_predictions)  # (n_models, total_samples, n_classes)
        
        # Compute ensemble statistics
        mean_probs = ensemble_predictions.mean(axis=0)
        predictions = mean_probs.argmax(axis=1)
        
        # Compute uncertainty metrics
        uncertainty_metrics = self._compute_ensemble_uncertainty(ensemble_predictions)
        
        return {
            'predictions': predictions,
            'targets': np.array(all_targets),
            'ensemble_predictions': ensemble_predictions,
            'uncertainty_metrics': uncertainty_metrics
        }
    
    def _compute_ensemble_uncertainty(self, ensemble_predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute uncertainty metrics from ensemble predictions."""
        # ensemble_predictions shape: (n_models, total_samples, n_classes)
        
        mean_probs = ensemble_predictions.mean(axis=0)
        
        # Predictive entropy
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        
        # Variance across ensemble
        ensemble_variance = ensemble_predictions.var(axis=0).max(axis=1)
        
        # Disagreement (standard deviation of predictions)
        prediction_std = ensemble_predictions.std(axis=0).max(axis=1)
        
        # Confidence interval width (95%)
        percentile_2_5 = np.percentile(ensemble_predictions.max(axis=2), 2.5, axis=0)
        percentile_97_5 = np.percentile(ensemble_predictions.max(axis=2), 97.5, axis=0)
        confidence_interval_width = percentile_97_5 - percentile_2_5
        
        return {
            'predictive_entropy': predictive_entropy,
            'ensemble_variance': ensemble_variance,
            'prediction_std': prediction_std,
            'confidence_interval_width': confidence_interval_width,
            'mean_probs': mean_probs
        }


class UncertaintyCalibration:
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, targets: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """Fit temperature scaling for calibration."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            loss = nn.CrossEntropyLoss()(logits / self.temperature, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.is_fitted = True
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            raise ValueError("Temperature scaling must be fitted before prediction")
        return logits / self.temperature


class BayesianWrapper:
    """Wrapper to convert deterministic models to Bayesian using variational inference."""
    
    def __init__(self, base_model: nn.Module, prior_std: float = 1.0, device: str = 'cpu'):
        self.base_model = base_model
        self.prior_std = prior_std
        self.device = device
        self.variational_params = {}
        
        self._convert_to_bayesian()
    
    def _convert_to_bayesian(self):
        """Convert linear and conv layers to Bayesian layers."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Store original parameters
                weight_shape = module.weight.shape
                if hasattr(module, 'bias') and module.bias is not None:
                    bias_shape = module.bias.shape
                    has_bias = True
                else:
                    has_bias = False
                
                # Create variational parameters
                self.variational_params[f'{name}_weight_mean'] = nn.Parameter(module.weight.clone())
                self.variational_params[f'{name}_weight_logvar'] = nn.Parameter(
                    torch.full(weight_shape, -3.0)  # Initialize with small variance
                )
                
                if has_bias:
                    self.variational_params[f'{name}_bias_mean'] = nn.Parameter(module.bias.clone())
                    self.variational_params[f'{name}_bias_logvar'] = nn.Parameter(
                        torch.full(bias_shape, -3.0)
                    )
    
    def sample_weights(self) -> Dict[str, torch.Tensor]:
        """Sample weights from variational distribution."""
        sampled_weights = {}
        
        for name, param in self.variational_params.items():
            if 'mean' in name:
                base_name = name.replace('_mean', '')
                logvar_name = name.replace('_mean', '_logvar')
                
                mean = param
                logvar = self.variational_params[logvar_name]
                std = torch.exp(0.5 * logvar)
                
                eps = torch.randn_like(mean)
                sampled_weights[base_name] = mean + std * eps
        
        return sampled_weights
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational and prior distributions."""
        kl = 0.0
        
        for name, param in self.variational_params.items():
            if 'mean' in name:
                base_name = name.replace('_mean', '')
                logvar_name = name.replace('_mean', '_logvar')
                
                mean = param
                logvar = self.variational_params[logvar_name]
                
                # KL divergence for Gaussian distributions
                kl += -0.5 * torch.sum(1 + logvar - (mean / self.prior_std).pow(2) - logvar.exp() / (self.prior_std ** 2))
        
        return kl


def train_ensemble(
    model_factory: Callable,
    train_loader,
    val_loader,
    n_models: int = 5,
    save_dir: str = 'ensemble_models',
    device: str = 'cpu',
    **training_kwargs
) -> List[str]:
    """Train an ensemble of models with different initializations."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    model_paths = []
    
    for i in range(n_models):
        print(f"Training ensemble model {i+1}/{n_models}")
        
        # Create model with different random initialization
        torch.manual_seed(42 + i)  # Different seed for each model
        model = model_factory()
        model.to(device)
        
        # Train model (this would need to be implemented)
        # trained_model = train_single_model(model, train_loader, val_loader, **training_kwargs)
        
        # Save model
        model_path = save_path / f"ensemble_model_{i}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': training_kwargs.get('model_config', {})
        }, model_path)
        
        model_paths.append(str(model_path))
    
    return model_paths


def evaluate_uncertainty_quality(
    uncertainty_scores: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Evaluate the quality of uncertainty estimates."""
    
    # Sort by uncertainty (ascending)
    sorted_indices = np.argsort(uncertainty_scores)
    sorted_correctness = correctness[sorted_indices]
    
    # Compute accuracy vs uncertainty correlation
    correlation = np.corrcoef(uncertainty_scores, correctness.astype(float))[0, 1]
    
    # Compute calibration metrics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    
    # Normalize uncertainty scores to [0, 1]
    normalized_uncertainty = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (normalized_uncertainty > bin_lower) & (normalized_uncertainty <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_uncertainty_in_bin = normalized_uncertainty[in_bin].mean()
            
            calibration_error = abs(avg_uncertainty_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            mce = max(mce, calibration_error)
    
    # Area Under the Risk-Coverage Curve (AURC)
    n_samples = len(uncertainty_scores)
    risks = []
    coverages = []
    
    for i in range(n_samples):
        coverage = (n_samples - i) / n_samples
        remaining_samples = sorted_indices[i:]
        risk = 1 - sorted_correctness[i:].mean() if len(remaining_samples) > 0 else 0
        
        coverages.append(coverage)
        risks.append(risk)
    
    aurc = np.trapz(risks, coverages)
    
    return {
        'uncertainty_accuracy_correlation': correlation if not np.isnan(correlation) else 0.0,
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'area_under_risk_coverage_curve': aurc
    }