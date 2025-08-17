#!/usr/bin/env python3
"""
Evaluation script for MRI classification and segmentation models.

Usage:
    python evaluate.py --config configs/base_config.yaml --checkpoint experiments/base_experiment/best_model.pth
    python evaluate.py --config configs/efficientnet_focal.yaml --checkpoint experiments/efficientnet_focal/best_model.pth --uncertainty
    python evaluate.py --config configs/segmentation_unet.yaml --checkpoint experiments/segmentation_unet/best_model.pth --visualize
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.data.dataset import create_dataloaders
from src.models.classification_2d import create_model
from src.models.classification_3d import create_3d_model
from src.models.segmentation import create_segmentation_model
from src.metrics.metrics import MetricsCalculator, SegmentationMetricsCalculator, UncertaintyMetrics
from src.utils.uncertainty import MCDropoutPredictor, DeepEnsemble
from src.visualization.explainability import (
    GradCAM, GradCAMPlusPlus, IntegratedGradients, ExplainabilityVisualizer,
    find_conv_layers
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MRI classification/segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--uncertainty', action='store_true', help='Evaluate uncertainty quantification')
    parser.add_argument('--explainability', action='store_true', help='Generate explainability visualizations')
    parser.add_argument('--visualize', action='store_true', help='Generate result visualizations')
    parser.add_argument('--save-predictions', action='store_true', help='Save model predictions')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def evaluate_classification(model, data_loader, device, use_uncertainty=False, mc_samples=100):
    """Evaluate classification model."""
    metrics_calc = MetricsCalculator(device=device)
    uncertainty_calc = UncertaintyMetrics()
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    if use_uncertainty:
        mc_predictor = MCDropoutPredictor(model, n_samples=mc_samples, device=device)
        uncertainty_results = mc_predictor.predict_with_uncertainty(data_loader)
        
        metrics = metrics_calc.compute_classification_metrics()
        uncertainty_metrics = uncertainty_calc.compute_uncertainty_metrics()
        
        return {
            'classification_metrics': metrics,
            'uncertainty_metrics': uncertainty_metrics,
            'uncertainty_results': uncertainty_results
        }
    else:
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                metrics_calc.update(outputs, targets, probabilities)
        
        metrics = metrics_calc.compute_classification_metrics()
        
        return {
            'classification_metrics': metrics,
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'targets': np.array(all_targets)
        }


def evaluate_segmentation(model, data_loader, device):
    """Evaluate segmentation model."""
    metrics_calc = SegmentationMetricsCalculator(device=device)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            metrics_calc.update(outputs, targets)
    
    metrics = metrics_calc.compute_segmentation_metrics()
    
    return {
        'segmentation_metrics': metrics,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }


def generate_explainability_maps(model, data_loader, device, config, output_dir):
    """Generate explainability visualizations."""
    print("Generating explainability maps...")
    
    # Find convolutional layers for GradCAM
    conv_layers = find_conv_layers(model)
    if not conv_layers:
        print("No convolutional layers found for GradCAM")
        return
    
    # Use the last few conv layers
    target_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
    
    # Initialize explainability methods
    gradcam = GradCAM(model, target_layers, device)
    gradcam_plus = GradCAMPlusPlus(model, target_layers, device)
    integrated_gradients = IntegratedGradients(model, device)
    
    # Initialize visualizer
    visualizer = ExplainabilityVisualizer(save_dir=output_dir / 'explainability')
    
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    # Generate explanations for a few samples
    model.eval()
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= 10:  # Only process first 10 batches
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        for j in range(min(2, inputs.size(0))):  # Process first 2 samples in batch
            sample_input = inputs[j:j+1]
            sample_target = targets[j].item()
            
            # Get prediction
            with torch.no_grad():
                output = model(sample_input)
                if isinstance(output, tuple):
                    output = output[0]
                predicted_class = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
            
            # Generate different explanations
            explanations = {}
            
            try:
                # GradCAM
                gradcam_map = gradcam.generate_cam(sample_input, predicted_class, target_layers[-1])
                explanations['GradCAM'] = gradcam_map
                
                # GradCAM++
                gradcam_plus_map = gradcam_plus.generate_cam(sample_input, predicted_class, target_layers[-1])
                explanations['GradCAM++'] = gradcam_plus_map
                
                # Integrated Gradients
                ig_map = integrated_gradients.generate_attribution(sample_input, predicted_class)
                explanations['Integrated Gradients'] = np.abs(ig_map).max(axis=0)  # Take max across channels
                
            except Exception as e:
                print(f"Error generating explanations for sample {i}_{j}: {e}")
                continue
            
            # Prepare original image for visualization
            original_image = sample_input.squeeze().cpu().numpy()
            if original_image.ndim == 3:  # Multi-channel
                original_image = np.transpose(original_image, (1, 2, 0))
                if original_image.shape[2] == 3:  # RGB
                    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
                else:
                    original_image = original_image[:, :, 0]  # Take first channel
            
            # Create visualization
            title = f"Sample {i}_{j} - True: {class_names[sample_target]}, Pred: {class_names[predicted_class]}"
            save_name = f"explanation_sample_{i}_{j}"
            
            visualizer.plot_multiple_explanations(
                original_image=original_image,
                explanations=explanations,
                title=title,
                save_name=save_name
            )
    
    # Cleanup
    gradcam.cleanup()
    gradcam_plus.cleanup()
    
    print(f"Explainability maps saved to: {output_dir / 'explainability'}")


def save_results(results, output_dir, config):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = output_dir / 'metrics.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    json_results[key][k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_results[key][k] = float(v)
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    with open(metrics_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {metrics_file}")
    
    # Save detailed report
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write("MRI Classification/Segmentation Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Task: {config['training'].get('task_type', 'classification')}\n\n")
        
        if 'classification_metrics' in results:
            f.write("Classification Metrics:\n")
            f.write("-" * 20 + "\n")
            metrics = results['classification_metrics']
            f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"F1 Score (macro): {metrics.get('f1_macro', 0):.4f}\n")
            f.write(f"AUC (macro): {metrics.get('auc_macro', 0):.4f}\n")
            f.write(f"Sensitivity @ 95% Specificity: {metrics.get('sensitivity_at_95_specificity_macro', 0):.4f}\n\n")
        
        if 'segmentation_metrics' in results:
            f.write("Segmentation Metrics:\n")
            f.write("-" * 20 + "\n")
            metrics = results['segmentation_metrics']
            f.write(f"Dice Score: {metrics.get('dice_mean', 0):.4f} ± {metrics.get('dice_std', 0):.4f}\n")
            f.write(f"IoU Score: {metrics.get('iou_mean', 0):.4f} ± {metrics.get('iou_std', 0):.4f}\n")
            f.write(f"Hausdorff95: {metrics.get('hausdorff95', 0):.4f}\n\n")
        
        if 'uncertainty_metrics' in results:
            f.write("Uncertainty Metrics:\n")
            f.write("-" * 20 + "\n")
            metrics = results['uncertainty_metrics']
            f.write(f"Expected Calibration Error: {metrics.get('ece', 0):.4f}\n")
            f.write(f"Predictive Entropy: {metrics.get('entropy_mean', 0):.4f} ± {metrics.get('entropy_std', 0):.4f}\n")
            f.write(f"Confidence: {metrics.get('confidence_mean', 0):.4f} ± {metrics.get('confidence_std', 0):.4f}\n")
    
    print(f"Detailed report saved to: {report_file}")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        use_weighted_sampling=False  # No weighted sampling for evaluation
    )
    
    val_loader = data_loaders['val']
    
    # Create and load model
    print(f"Creating model: {config['model']['name']}")
    model = create_model_from_config(config)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Determine task type
    task_type = config['training'].get('task_type', 'classification')
    
    # Evaluate model
    print(f"Evaluating {task_type} model...")
    if task_type == 'classification':
        results = evaluate_classification(
            model=model,
            data_loader=val_loader,
            device=device,
            use_uncertainty=args.uncertainty,
            mc_samples=config['evaluation'].get('mc_samples', 100)
        )
    else:  # segmentation
        results = evaluate_segmentation(
            model=model,
            data_loader=val_loader,
            device=device
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results(results, output_dir, config)
    
    # Generate explainability maps if requested
    if args.explainability and task_type == 'classification':
        generate_explainability_maps(model, val_loader, device, config, output_dir)
    
    # Save predictions if requested
    if args.save_predictions:
        if 'predictions' in results:
            np.save(output_dir / 'predictions.npy', results['predictions'])
            np.save(output_dir / 'targets.npy', results['targets'])
            if 'probabilities' in results:
                np.save(output_dir / 'probabilities.npy', results['probabilities'])
            print(f"Predictions saved to: {output_dir}")
    
    print("Evaluation completed!")
    
    # Print summary
    if task_type == 'classification':
        metrics = results['classification_metrics']
        print(f"\nClassification Results:")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"F1 Score (macro): {metrics.get('f1_macro', 0):.4f}")
        print(f"AUC (macro): {metrics.get('auc_macro', 0):.4f}")
    else:
        metrics = results['segmentation_metrics']
        print(f"\nSegmentation Results:")
        print(f"Dice Score: {metrics.get('dice_mean', 0):.4f}")
        print(f"IoU Score: {metrics.get('iou_mean', 0):.4f}")


if __name__ == '__main__':
    main()