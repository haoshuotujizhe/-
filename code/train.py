import torch
import torch.nn as nn
import os
import time
import json
import torch.optim as optim
from tqdm import tqdm
import sys
# 兼容新旧 AMP API
try:
    from torch.amp import autocast, GradScaler
    AMP_HAS_DEVICE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_HAS_DEVICE = False


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import build_model
from utils import get_dataloaders, save_model, set_seed

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda",dtype=amp_dtype,enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def validate(model, dataloader, criterion, device,use_amp=False):
    model.eval()  # 切换模型到评估模式（关闭 Dropout、固定 BN 层统计量）
    total_loss, total_correct, total = 0, 0, 0  # 累计损失、正确数、总样本数
    
    # 禁用梯度计算（验证阶段不需要反向传播，节省内存和计算资源）
    with torch.no_grad():
        # 遍历验证数据加载器
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)  # 数据移到设备
            with autocast(device_type="cuda",enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
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
    use_amp = device.type == "cuda"
    # 根据设备选择 dtype（Ampere起优先用 bf16）
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    #scaler = GradScaler('cuda',enabled=use_amp)
    if AMP_HAS_DEVICE:
        scaler = GradScaler(enabled=use_amp, device='cuda' if device.type == 'cuda' else 'cpu')
    else:
        scaler = GradScaler('cuda',enabled=use_amp)
    
    # 数据加载
    train_loader, val_loader = get_dataloaders(
        train_dir=config["train_dir"],
        train_label_csv=config["train_label_csv"],
        val_dir=config["val_dir"],
        config=config
    )

    # 模型与优化
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # 训练循环
    best_acc = 0.0
    patience_counter = 0
    max_patience = 7
    model_save_dir = os.path.join(parent_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_model.pth")

    for epoch in range(config["epochs"]):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}] LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_model(model, best_model_path)
            print("✅ Saved new best model!")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter} epochs")

        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break

    print("Training complete! Best accuracy:", best_acc)