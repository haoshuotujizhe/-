import torch        #采用pytorch训练
import torch.nn as nn       #导入pytorchnn，用于快速构建神经网络
from torchvision import models      #导入pytorch官方给的模型库

from torchvision.models import EfficientNet_B7_Weights

# model=models.efficientnet_b7(pretrained=True)       #导入efficientnet_b7预训练模型

class Mymodel(nn.Module):        
    def __init__(self, num_classes=100, use_pretrained=True):
        super(Mymodel, self).__init__()
        
        if use_pretrained:
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b7(weights=weights)
        else:
            self.backbone = models.efficientnet_b7(weights=None)
        
        in_features = self.backbone.classifier[1].in_features
        
        # ✅ 使用简单的分类头（不要过度复杂化）
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def build_model(config):
    num_classes = config.get("num_classes", 100)
    use_pretrained = config.get("use_pretrained", True)
    model = Mymodel(num_classes=num_classes, use_pretrained=use_pretrained)
    return model






