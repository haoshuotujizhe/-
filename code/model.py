import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    ConvNeXt_Base_Weights
)


class Mymodel(nn.Module):
    def __init__(self, num_classes=152, use_pretrained=True, model_name="convnext_base", stochastic_depth=0.0):
        super(Mymodel, self).__init__()
        self.model_name = model_name
        
        if model_name == "efficientnet_b6":
            if use_pretrained:
                self.backbone = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b6(weights=None)
            
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.45, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        
        elif model_name == "convnext_base":
            if use_pretrained:
                self.backbone = models.convnext_base(
                    weights=ConvNeXt_Base_Weights.IMAGENET1K_V1,
                    stochastic_depth_prob=stochastic_depth
                )
            else:
                self.backbone = models.convnext_base(
                    weights=None,
                    stochastic_depth_prob=stochastic_depth
                )
            
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(in_features),
                nn.Linear(in_features, in_features // 2),
                nn.GELU(),
                nn.Dropout(p=0.35),
                nn.Linear(in_features // 2, num_classes)
            )
            
        elif model_name == "efficientnet_b7":
            if use_pretrained:
                self.backbone = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b7(weights=None)
            
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)


def build_model(config):
    return Mymodel(
        num_classes=config.get("num_classes", 152),
        use_pretrained=config.get("use_pretrained", True),
        model_name=config.get("model_name", "convnext_base"),
        stochastic_depth=config.get("stochastic_depth", 0.0)
    )