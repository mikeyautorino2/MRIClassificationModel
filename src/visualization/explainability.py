import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns


class GradCAM:
    def __init__(self, model: nn.Module, target_layers: List[str], device: str = 'cpu'):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate Class Activation Map."""
        
        if layer_name is None:
            layer_name = self.target_layers[-1]  # Use last layer by default
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]  # Take logits if attention weights are returned
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[layer_name][0]  # (C, H, W) or (C, D, H, W)
        activations = self.activations[layer_name][0]  # (C, H, W) or (C, D, H, W)
        
        # Compute weights
        if gradients.dim() == 4:  # 3D case (C, D, H, W)
            weights = torch.mean(gradients, dim=(1, 2, 3))
        else:  # 2D case (C, H, W)
            weights = torch.mean(gradients, dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


class GradCAMPlusPlus(GradCAM):
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ activation map."""
        
        if layer_name is None:
            layer_name = self.target_layers[-1]
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[layer_name][0]
        activations = self.activations[layer_name][0]
        
        # Compute alpha weights (Grad-CAM++ improvement)
        if gradients.dim() == 4:  # 3D case
            alpha_num = gradients.pow(2)
            alpha_denom = 2.0 * gradients.pow(2) + activations.mul(gradients.pow(3)).sum(dim=(1, 2, 3), keepdim=True)
            alpha = alpha_num / (alpha_denom + 1e-7)
            weights = (alpha * F.relu(gradients)).sum(dim=(1, 2, 3))
        else:  # 2D case
            alpha_num = gradients.pow(2)
            alpha_denom = 2.0 * gradients.pow(2) + activations.mul(gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
            alpha = alpha_num / (alpha_denom + 1e-7)
            weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class IntegratedGradients:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """Generate Integrated Gradients attribution map."""
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # Forward pass to get target class
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps, device=self.device)
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolated input
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward and backward pass
            output = self.model(interpolated_input)
            if isinstance(output, tuple):
                output = output[0]
            
            self.model.zero_grad()
            target = output[0, target_class]
            target.backward()
            
            # Accumulate gradients
            integrated_gradients += interpolated_input.grad / steps
        
        # Final attribution
        attribution = (input_tensor - baseline) * integrated_gradients
        attribution = attribution.squeeze().cpu().numpy()
        
        return attribution


class LIME:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def explain_instance(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 1000,
        num_features: int = 100,
        superpixel_fn=None
    ) -> np.ndarray:
        """Generate LIME explanation for an instance."""
        
        # This is a simplified LIME implementation
        # In practice, you'd use the actual LIME library
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction for original image
        with torch.no_grad():
            original_output = self.model(input_tensor)
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
        
        # Generate perturbations and explanations
        # This is a placeholder - real LIME would use proper superpixel segmentation
        perturbations = []
        predictions = []
        
        for _ in range(num_samples):
            # Create random mask
            mask = torch.rand_like(input_tensor) > 0.5
            perturbed_input = input_tensor * mask.float()
            
            with torch.no_grad():
                output = self.model(perturbed_input)
                if isinstance(output, tuple):
                    output = output[0]
                prob = F.softmax(output, dim=1)[0, target_class].item()
            
            perturbations.append(mask.cpu().numpy().flatten())
            predictions.append(prob)
        
        # Simple linear regression to find important features
        from sklearn.linear_model import LinearRegression
        
        X = np.array(perturbations)
        y = np.array(predictions)
        
        reg = LinearRegression().fit(X, y)
        importance = reg.coef_.reshape(input_tensor.shape[1:])
        
        return importance


class SaliencyMap:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def generate_saliency(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate vanilla gradient saliency map."""
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Get gradients
        saliency = input_tensor.grad.abs().max(dim=1)[0]  # Max across channels
        saliency = saliency.squeeze().cpu().numpy()
        
        # Normalize
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency


class ExplainabilityVisualizer:
    def __init__(self, save_dir: str = 'explanations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        
        # Normalize image to 0-255
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Resize heatmap to match image
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cm.get_cmap(colormap)(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def plot_explanation(
        self,
        original_image: np.ndarray,
        explanation_map: np.ndarray,
        title: str = 'Explanation',
        save_name: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        predicted_class: Optional[int] = None,
        confidence: Optional[float] = None
    ):
        """Plot and save explanation visualization."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(original_image.shape) == 3:
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Explanation heatmap
        im = axes[1].imshow(explanation_map, cmap='hot')
        axes[1].set_title('Explanation Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        overlay = self.overlay_heatmap(original_image, explanation_map)
        axes[2].imshow(overlay)
        
        # Add prediction info to title
        overlay_title = 'Overlay'
        if predicted_class is not None:
            if class_names is not None:
                overlay_title += f'\nPredicted: {class_names[predicted_class]}'
            else:
                overlay_title += f'\nPredicted: Class {predicted_class}'
        if confidence is not None:
            overlay_title += f'\nConfidence: {confidence:.3f}'
        
        axes[2].set_title(overlay_title)
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_explanations(
        self,
        original_image: np.ndarray,
        explanations: Dict[str, np.ndarray],
        title: str = 'Multiple Explanations',
        save_name: Optional[str] = None
    ):
        """Plot multiple explanation methods side by side."""
        
        n_methods = len(explanations)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        if n_methods == 0:
            return fig
        
        # Original image (spans both rows)
        axes[0, 0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Explanation methods
        for i, (method_name, explanation) in enumerate(explanations.items(), 1):
            # Heatmap
            im = axes[0, i].imshow(explanation, cmap='hot')
            axes[0, i].set_title(f'{method_name}\nHeatmap')
            axes[0, i].axis('off')
            
            # Overlay
            overlay = self.overlay_heatmap(original_image, explanation)
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'{method_name}\nOverlay')
            axes[1, i].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_overlay(
        self,
        original_image: np.ndarray,
        prediction_map: np.ndarray,
        uncertainty_map: np.ndarray,
        segmentation_mask: Optional[np.ndarray] = None,
        title: str = 'Uncertainty Visualization',
        save_name: Optional[str] = None
    ):
        """Plot prediction with uncertainty overlay."""
        
        n_plots = 4 if segmentation_mask is not None else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction
        im1 = axes[1].imshow(prediction_map, cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Uncertainty
        im2 = axes[2].imshow(uncertainty_map, cmap='hot')
        axes[2].set_title('Uncertainty')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        # Segmentation overlay if provided
        if segmentation_mask is not None:
            overlay = self.overlay_heatmap(original_image, segmentation_mask, alpha=0.3, colormap='viridis')
            axes[3].imshow(overlay)
            axes[3].set_title('Segmentation\nOverlay')
            axes[3].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig


def get_model_layers(model: nn.Module) -> List[str]:
    """Get list of all layer names in the model."""
    return [name for name, _ in model.named_modules()]


def find_conv_layers(model: nn.Module) -> List[str]:
    """Find all convolutional layer names in the model."""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            conv_layers.append(name)
    return conv_layers