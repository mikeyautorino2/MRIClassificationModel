import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, Dict, Any


class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return F.dropout(x, self.p, training=True)


class AttentionPooling(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_slices, features)
        attention_weights = self.attention(x)  # (batch_size, num_slices, 1)
        weighted_features = x * attention_weights  # (batch_size, num_slices, features)
        aggregated = torch.sum(weighted_features, dim=1)  # (batch_size, features)
        return aggregated, attention_weights.squeeze(-1)


class EfficientNetB0Classifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 4, 
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            num_classes=0  # Remove classification head
        )
        
        in_features = self.backbone.num_features
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class ResNet50Classifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 4, 
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classification head
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        return self.backbone(x)


class SliceAggregationClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        num_classes: int = 4,
        aggregation_method: str = 'mean',  # 'mean', 'attention', 'max'
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_mc_dropout: bool = False
    ):
        super().__init__()
        
        self.aggregation_method = aggregation_method
        
        if backbone == 'efficientnet_b0':
            self.feature_extractor = EfficientNetB0Classifier(
                num_classes=0, 
                pretrained=pretrained,
                dropout_rate=dropout_rate,
                use_mc_dropout=use_mc_dropout
            )
            in_features = 1280  # EfficientNet-B0 feature size
        elif backbone == 'resnet50':
            self.feature_extractor = ResNet50Classifier(
                num_classes=0,
                pretrained=pretrained,
                dropout_rate=dropout_rate,
                use_mc_dropout=use_mc_dropout
            )
            in_features = 2048  # ResNet50 feature size
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classifier from feature extractor
        self.feature_extractor.classifier = nn.Identity()
        
        if aggregation_method == 'attention':
            self.attention_pooling = AttentionPooling(in_features)
        
        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = MCDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_slices, channels, height, width)
        batch_size, num_slices = x.shape[:2]
        
        # Reshape to process all slices at once
        x = x.view(-1, *x.shape[2:])  # (batch_size * num_slices, channels, height, width)
        
        # Extract features from all slices
        slice_features = self.feature_extractor.get_features(x)  # (batch_size * num_slices, features)
        
        # Reshape back to separate slices
        slice_features = slice_features.view(batch_size, num_slices, -1)  # (batch_size, num_slices, features)
        
        # Aggregate slice features
        if self.aggregation_method == 'mean':
            aggregated_features = torch.mean(slice_features, dim=1)
            attention_weights = None
        elif self.aggregation_method == 'max':
            aggregated_features, _ = torch.max(slice_features, dim=1)
            attention_weights = None
        elif self.aggregation_method == 'attention':
            aggregated_features, attention_weights = self.attention_pooling(slice_features)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        # Final classification
        aggregated_features = self.dropout(aggregated_features)
        logits = self.classifier(aggregated_features)
        
        if attention_weights is not None:
            return logits, attention_weights
        return logits


class EnsembleClassifier(nn.Module):
    def __init__(self, models: list, ensemble_method: str = 'mean'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]  # Take logits if attention weights are returned
                outputs.append(output)
        
        if self.ensemble_method == 'mean':
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.ensemble_method == 'voting':
            predictions = torch.stack([F.softmax(out, dim=-1) for out in outputs])
            return torch.mean(predictions, dim=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    use_mc_dropout: bool = False,
    **kwargs
) -> nn.Module:
    
    if model_name == 'efficientnet_b0':
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    elif model_name == 'resnet50':
        return ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    elif model_name in ['slice_efficientnet_b0', 'slice_resnet50']:
        backbone = model_name.replace('slice_', '')
        return SliceAggregationClassifier(
            backbone=backbone,
            num_classes=num_classes,
            aggregation_method=kwargs.get('aggregation_method', 'mean'),
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def freeze_backbone(model: nn.Module, freeze: bool = True):
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
    elif hasattr(model, 'feature_extractor'):
        for param in model.feature_extractor.parameters():
            param.requires_grad = not freeze