import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import Optional, Dict, Any


class MCDropout3D(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return F.dropout3d(x, self.p, training=True)


class VideoResNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        
        # Use Video ResNet from torchvision
        self.backbone = models.video.r3d_18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout3D(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


class R3D18Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        
        self.backbone = models.video.r3d_18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout3D(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


class EfficientNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0
    ):
        super().__init__()
        
        self.use_mc_dropout = use_mc_dropout
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks - simplified 3D version
        self.blocks = nn.ModuleList([
            self._make_mbconv_block(32, 64, 3, 1, 1),
            self._make_mbconv_block(64, 128, 3, 2, 2),
            self._make_mbconv_block(128, 256, 5, 2, 2),
            self._make_mbconv_block(256, 512, 3, 2, 3),
            self._make_mbconv_block(512, 512, 5, 1, 3),
            self._make_mbconv_block(512, 1024, 5, 2, 4),
            self._make_mbconv_block(1024, 1024, 3, 1, 1),
        ])
        
        # Head
        self.head_conv = nn.Sequential(
            nn.Conv3d(1024, 1280, kernel_size=1, bias=False),
            nn.BatchNorm3d(1280),
            nn.SiLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        if use_mc_dropout:
            self.dropout = MCDropout3D(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(1280, num_classes)
        
    def _make_mbconv_block(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        blocks = []
        for i in range(num_blocks):
            blocks.append(MBConvBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                stride if i == 0 else 1
            ))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=6):
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze and Excitation
        layers.append(SqueezeExcitation3D(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class DenseNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        
        # First convolution
        self.features = nn.Sequential(
            nn.Conv3d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition3D(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout3D(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, dropout_rate):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate,
                dropout_rate
            )
            self.add_module(f'denselayer{i+1}', layer)
    
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer3D(nn.Module):
    def __init__(self, num_input_features, growth_rate, dropout_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, 4 * growth_rate,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(4 * growth_rate, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        new_features = super().forward(x)
        if self.dropout_rate > 0:
            new_features = F.dropout(new_features, p=self.dropout_rate, training=self.training)
        return new_features


class Transition3D(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                         kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


def create_3d_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    use_mc_dropout: bool = False,
    **kwargs
) -> nn.Module:
    
    if model_name == 'video_resnet3d' or model_name == 'videoresnet3d':
        return VideoResNet3D(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    elif model_name == 'r3d_18':
        return R3D18Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    elif model_name == 'efficientnet3d':
        return EfficientNet3D(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    elif model_name == 'densenet3d':
        return DenseNet3D(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    else:
        raise ValueError(f"Unsupported 3D model: {model_name}")


def freeze_3d_backbone(model: nn.Module, freeze: bool = True):
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
    elif hasattr(model, 'features'):
        for param in model.features.parameters():
            param.requires_grad = not freeze