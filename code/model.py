import torch        #采用pytorch训练
import torch.nn as nn       #导入pytorchnn，用于快速构建神经网络
from torchvision import models      #导入pytorch官方给的模型库

from torchvision.models import EfficientNet_B7_Weights

# model=models.efficientnet_b7(pretrained=True)       #导入efficientnet_b7预训练模型

class Mymodel(nn.Module):        
    def __init__(self,num_classes=100,use_pretrained=True):
        super(Mymodel,self).__init__()      #super超类；调用 nn.Module 的构造方法，初始化 nn.Module 中定义的核心属性
        if use_pretrained:
            self.backbone=models.efficientnet_b7(weights=EfficientNet_B7_Weights)
        else:
            self.backbone=models.efficientnet_b7(weights=None)
        
        in_features=self.backbone.classifier[1].in_features        #获取最后一层全连接层（fc）的输入特征数
        self.backbone.classifier[1]=nn.Linear(in_features,num_classes)     #输入维度，输出维度100种花朵
    
        
    def forward(self,x):        #x是输入模型的张量（通常是图像数据，形状为[batch_size, channels, height, width]）
        return self.backbone(x)
def build_model(config):
    num_classes = config.get("num_classes", 100)
    use_pretrained = config.get("use_pretrained", True)
    model = Mymodel(num_classes=num_classes, use_pretrained=use_pretrained)
    return model
    
    
    
        


