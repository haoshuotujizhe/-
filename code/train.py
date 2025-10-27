import torch
import torch.nn as nn
import os
import time
import json
import torch.optim as optim
from tqdm import tqdm
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import build_model
from utils import get_dataloaders, save_model, set_seed

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # 切换模型到训练模式（启用 Dropout、BN 层训练模式）
    total_loss, total_correct, total = 0, 0, 0  # 累计损失、正确数、总样本数
    
    # 遍历训练数据加载器，tqdm 显示进度条
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        # 将数据移到训练设备（GPU/CPU）
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 梯度清零（避免上一轮梯度累积）
        optimizer.zero_grad()
        
        # 前向传播：模型输出预测结果
        outputs = model(imgs)
        
        # 计算损失（预测结果与真实标签的差异）
        loss = criterion(outputs, labels)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 优化器更新模型参数
        optimizer.step()
        
        # 累计损失（乘以批次大小，因为 loss 是批次平均损失）
        total_loss += loss.item() * imgs.size(0)
        
        # 计算预测准确率：取输出中概率最大的类别作为预测结果
        _, preds = torch.max(outputs, 1)  # preds 是预测的类别索引
        total_correct += (preds == labels).sum().item()  # 统计正确预测的数量
        total += labels.size(0)  # 累计总样本数
    
    # 返回当前 epoch 的平均损失和准确率
    return total_loss / total, total_correct / total


def validate(model, dataloader, criterion, device):
    model.eval()  # 切换模型到评估模式（关闭 Dropout、固定 BN 层统计量）
    total_loss, total_correct, total = 0, 0, 0  # 累计损失、正确数、总样本数
    
    # 禁用梯度计算（验证阶段不需要反向传播，节省内存和计算资源）
    with torch.no_grad():
        # 遍历验证数据加载器
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)  # 数据移到设备
            
            outputs = model(imgs)  # 前向传播（仅计算预测结果，不跟踪梯度）
            loss = criterion(outputs, labels)  # 计算验证损失
            
            # 累计损失和正确数（逻辑同训练阶段）
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # 返回验证集的平均损失和准确率
    return total_loss / total, total_correct / total


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))  # 例如：/xxx/xxx/code
# 回到 code 的上一级目录（即 code 和 model 的共同父目录）
    parent_dir = os.path.dirname(code_dir)  # 例如：/xxx/xxx
# 拼接出 config.json 的绝对路径（父目录 -> model -> config.json）
    config_path = os.path.join(parent_dir, "model", "config.json")  # 例如：/xxx/xxx/model/config.json
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 加载配置
    #with open("model/config.json", "r") as f:
    #    config = json.load(f)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, val_loader = get_dataloaders(
        train_dir=config["train_dir"],
        train_label_csv=config["train_label_csv"],
        val_dir=config["val_dir"],
        config=config
    )

    # 模型
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练循环
    best_acc = 0.0
    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("model", exist_ok=True)
            save_model(model, "/root/autodl-tmp/submission/model/best_model.pth")
            print("✅ Saved new best model!")

    print("Training complete! Best accuracy:", best_acc)