# MRI Brain Tumor Classification & Segmentation

A comprehensive deep learning pipeline for MRI brain tumor classification and segmentation with uncertainty quantification and explainability features.

## 🎯 Features

### **Task A - Classification**
- **2D Transfer Learning**: EfficientNet-B0, ResNet50 with slice aggregation (mean/attention)
- **3D Models**: VideoResNet3D, r3d_18, EfficientNet3D for volumetric context
- **Loss Functions**: Weighted Cross-Entropy, Focal Loss for class imbalance
- **Metrics**: AUC, F1, sensitivity @ 95% specificity

### **Task B - Segmentation**
- **Baseline**: U-Net (2D) → nnU-Net (3D) architectures
- **Advanced**: Attention U-Net, ResU-Net variants
- **Loss Functions**: Dice + BCE, Dice + Focal
- **Metrics**: Dice, IoU, Hausdorff95

### **Uncertainty & Explainability**
- **Uncertainty**: MC Dropout, Deep Ensembles (n=5) → predictive entropy
- **Explainability**: Grad-CAM/Grad-CAM++ highlighting suspicious regions
- **Visualization**: Overlay onto MRI + segmentation masks

### **Training Tricks**
- **Heavy Augmentation**: Random flip/rotate, elastic deformations, gamma/intensity shift
- **Advanced Techniques**: CutMix, MixUp, RandAugment (2D)
- **Optimization**: Early stopping on val Dice/AUC, Cosine LR with warmup, AMP (fp16)

## 📊 Dataset

The dataset contains **3,264 brain MRI images** organized into:

- **glioma_tumor**: 826 training, 100 testing images
- **meningioma_tumor**: 822 training, 115 testing images  
- **no_tumor**: 395 training, 105 testing images
- **pituitary_tumor**: 827 training, 74 testing images

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MRIClassification

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic 2D classification with EfficientNet-B0
python train.py --config configs/base_config.yaml

# 2D classification with Focal Loss and uncertainty
python train.py --config configs/efficientnet_focal.yaml --wandb

# 3D volumetric analysis with ResNet3D
python train.py --config configs/resnet_3d.yaml --gpu 0

# U-Net segmentation
python train.py --config configs/segmentation_unet.yaml

# Resume training from checkpoint
python train.py --config configs/base_config.yaml --resume experiments/base_experiment/checkpoint_epoch_50.pth
```

### Evaluation

```bash
# Basic evaluation
python evaluate.py --config configs/base_config.yaml --checkpoint experiments/base_experiment/best_model.pth

# Evaluation with uncertainty quantification
python evaluate.py --config configs/efficientnet_focal.yaml --checkpoint experiments/efficientnet_focal/best_model.pth --uncertainty

# Evaluation with explainability visualizations
python evaluate.py --config configs/base_config.yaml --checkpoint experiments/base_experiment/best_model.pth --explainability

# Save predictions and generate visualizations
python evaluate.py --config configs/segmentation_unet.yaml --checkpoint experiments/segmentation_unet/best_model.pth --save-predictions --visualize
```

## 📁 Project Structure

```
MRIClassification/
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   ├── efficientnet_focal.yaml # EfficientNet + Focal Loss
│   ├── resnet_3d.yaml         # 3D ResNet configuration
│   └── segmentation_unet.yaml # U-Net segmentation
├── data/                      # Dataset directory
│   ├── Training/              # Training images
│   └── Testing/               # Testing images
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures
│   │   ├── classification_2d.py
│   │   ├── classification_3d.py
│   │   └── segmentation.py
│   ├── losses/                # Loss functions
│   ├── metrics/               # Evaluation metrics
│   ├── utils/                 # Utilities (training, uncertainty)
│   └── visualization/         # Explainability tools
├── experiments/               # Training outputs
├── train.py                   # Main training script
├── evaluate.py               # Evaluation script
└── requirements.txt          # Dependencies
```

## ⚙️ Configuration

### Model Selection

```yaml
model:
  name: "efficientnet_b0"  # Options: efficientnet_b0, resnet50, r3d_18, unet2d
  num_classes: 4
  pretrained: true
  dropout_rate: 0.2
  use_mc_dropout: false
  aggregation_method: "attention"  # For slice models: mean, attention, max
```

### Loss Functions

```yaml
loss:
  name: "focal"  # Options: cross_entropy, weighted_ce, focal, dice_bce, combo
  params:
    alpha: [0.25, 0.75, 1.0, 0.5]  # Class weights
    gamma: 2.0
```

### Training Configuration

```yaml
training:
  num_epochs: 100
  use_amp: true          # Automatic Mixed Precision
  use_mixup: true        # MixUp augmentation
  use_cutmix: true       # CutMix augmentation
  early_stopping:
    enabled: true
    patience: 10
```

## 📈 Monitoring

The pipeline supports **Weights & Biases** for experiment tracking:

```bash
# Enable wandb logging
python train.py --config configs/base_config.yaml --wandb
```

## 🔬 Advanced Features

### Uncertainty Quantification

```python
# MC Dropout uncertainty
python evaluate.py --config configs/efficientnet_focal.yaml --checkpoint model.pth --uncertainty

# Deep Ensemble (train 5 models)
for i in range(5):
    python train.py --config configs/base_config.yaml --seed $((42 + i))
```

### Explainability

```python
# Generate Grad-CAM visualizations
python evaluate.py --config configs/base_config.yaml --checkpoint model.pth --explainability
```

### Custom Augmentations

The pipeline includes medical imaging specific augmentations:
- Elastic deformations
- Gamma/intensity shifts  
- Gaussian noise and blur
- Spatial transformations

## 🏥 Medical AI Best Practices

- **Class Imbalance**: Weighted sampling and focal loss
- **Validation**: Stratified splits respecting patient distribution
- **Uncertainty**: Confidence thresholds for clinical deployment
- **Explainability**: Grad-CAM overlays for radiologist review
- **Metrics**: Sensitivity @ 95% specificity for screening tasks

## 📊 Results

Expected performance ranges:

- **Classification AUC**: 0.85 - 0.95
- **F1 Score**: 0.80 - 0.90  
- **Segmentation Dice**: 0.75 - 0.90
- **Hausdorff95**: < 10 pixels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers for the MRI brain tumor dataset
- PyTorch and torchvision teams
- Medical imaging community for best practices
- Open source contributors