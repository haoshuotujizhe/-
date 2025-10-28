import torch
import torch.nn as nn
import os
import math
import random
import json
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy
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

def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(images, targets, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha):
    """返回 (images, (y_a, y_b, lam)) 或 (images, (targets, None, 1.0))"""
    if use_cutmix and cutmix_alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        batch_size, _, H, W = images.size()
        index = torch.randperm(batch_size, device=images.device)
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return images, (targets, targets[index], lam)
    if use_mixup and mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        mixed = lam * images + (1 - lam) * images[index, :]
        return mixed, (targets, targets[index], lam)
    return images, (targets, None, 1.0)

class ModelEMA:
    def __init__(self, model, decay=0.9998, device=None):
        self.ema = copy_model(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema = self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            if self.device is not None and self.device != p.device:
                p = p.to(self.device)
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)

def copy_model(model):
    import copy
    m = copy.deepcopy(model)
    return m

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, max_grad_norm=1.0, 
                    aug_cfg=None, ema=None, amp_dtype=torch.float16):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        original_labels = labels.clone()
        imgs, (y_a, y_b, lam) = apply_mixup_cutmix(
            imgs, labels,
            use_mixup=aug_cfg.get("use_mixup", False)if aug_cfg else False,
            mixup_alpha=aug_cfg.get("mixup_alpha", 0.0)if aug_cfg else 0.0,
            use_cutmix=aug_cfg.get("use_cutmix", False)if aug_cfg else False,
            cutmix_alpha=aug_cfg.get("cutmix_alpha", 0.0)if aug_cfg else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        if AMP_HAS_DEVICE:
            with autocast(device_type="cuda",dtype=amp_dtype,enabled=use_amp):
                outputs = model(imgs)
                if y_b is None:
                    loss = criterion(outputs, y_a)
                else:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            with autocast(enabled=use_amp):
                outputs = model(imgs)
                if y_b is None:
                    loss = criterion(outputs, y_a)
                else:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        # 注意：使用 Mixup/CutMix 时，此准确率为近似值（实际标签已混合），仅供参考
        total_correct += (preds == original_labels).sum().item()  
        total += labels.size(0)
    return total_loss / total, total_correct / total

def validate(model, dataloader, criterion, device,use_amp=False, use_tta=False, amp_dtype=torch.float16):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            # 简单TTA：原图 + 水平翻转
            if use_tta:
                imgs_flipped = torch.flip(imgs, dims=[3])
                if AMP_HAS_DEVICE:
                    with autocast(device_type="cuda",dtype=amp_dtype,enabled=use_amp):
                        out1 = model(imgs)
                        out2 = model(imgs_flipped)
                        outputs = 0.5 * (out1 + out2)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(enabled=use_amp):
                        out1 = model(imgs)
                        out2 = model(imgs_flipped)
                        outputs = 0.5 * (out1 + out2)
                        loss = criterion(outputs, labels)
            else:
                if AMP_HAS_DEVICE:
                    with autocast(device_type="cuda",dtype=amp_dtype,enabled=use_amp):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(enabled=use_amp):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total

def set_trainable(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def build_optimizer(model, lr_backbone, lr_head, weight_decay=1e-4):
    """判别式学习率 + 正确的权重衰减排除（bn/bias不衰减）"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    # 分类头参数单独分组（更大学习率）
    head_params = list(model.backbone.classifier.parameters())
    head_id = set([id(p) for p in head_params])

    decay_backbone = [p for p in decay if id(p) not in head_id]
    no_decay_backbone = [p for p in no_decay if id(p) not in head_id]
    decay_head = [p for p in decay if id(p) in head_id]
    no_decay_head = [p for p in no_decay if id(p) in head_id]

    param_groups = [
        {"params": decay_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": decay_head, "lr": lr_head, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": lr_head, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups)


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))  # 例如：/xxx/xxx/code
# 回到 code 的上一级目录（即 code 和 model 的共同父目录）
    parent_dir = os.path.dirname(code_dir)  # 例如：/xxx/xxx
# 拼接出 config.json 的绝对路径（父目录 -> model -> config.json）
    config_path = os.path.join(parent_dir, "model", "config.json")  # 例如：/xxx/xxx/model/config.json
    with open(config_path, "r") as f:
        config = json.load(f)
        
    #将配置中的相对路径转换为绝对路径
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(parent_dir, path))
    
    config["train_dir"] = resolve_path(config["train_dir"])
    config["val_dir"] = resolve_path(config["val_dir"])
    config["test_dir"] = resolve_path(config["test_dir"])
    config["train_label_csv"] = resolve_path(config["train_label_csv"])
    
    print("Resolved paths:")
    print(f"  train_dir: {config['train_dir']}")
    print(f"  val_dir: {config['val_dir']}")
    print(f"  train_label_csv: {config['train_label_csv']}")

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
        scaler = GradScaler(enabled=use_amp)
    
    # 数据加载
    train_loader, val_loader = get_dataloaders(
        train_dir=config["train_dir"],
        train_label_csv=config["train_label_csv"],
        val_dir=config["val_dir"],
        config=config
    )

    # 读取策略配置
    head_epochs = int(config.get("head_epochs", 0))
    lr_backbone = float(config.get("lr_backbone", config["learning_rate"]))
    lr_head = float(config.get("lr_head", config["learning_rate"]))
    use_tta = bool(config.get("use_tta", False))
    
    aug_cfg = {
        "use_mixup": bool(config.get("use_mixup", False)),
        "mixup_alpha": float(config.get("mixup_alpha", 0.0)),
        "use_cutmix": bool(config.get("use_cutmix", False)),
        "cutmix_alpha": float(config.get("cutmix_alpha", 0.0)),
    }
    
    # 模型
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # ✅ Phase 1 不初始化 EMA
    ema = None
    
    # 定义增强配置
    aug_cfg_phase1 = {
        "use_mixup": False,
        "mixup_alpha": 0.0,
        "use_cutmix": False,
        "cutmix_alpha": 0.0,
    }
    
    aug_cfg_phase2 = {
        "use_mixup": bool(config.get("use_mixup", False)),
        "mixup_alpha": float(config.get("mixup_alpha", 0.0)),
        "use_cutmix": bool(config.get("use_cutmix", False)),
        "cutmix_alpha": float(config.get("cutmix_alpha", 0.0)),
    }
    
    print(f"✅ 增强配置已设置:")
    print(f"   Phase 1 - Mixup: {aug_cfg_phase1['use_mixup']}, CutMix: {aug_cfg_phase1['use_cutmix']}")
    print(f"   Phase 2 - Mixup: {aug_cfg_phase2['use_mixup']}, CutMix: {aug_cfg_phase2['use_cutmix']}")
    print(f"   Label Smoothing: 0.05")

    # 阶段1：只训练分类头
    if head_epochs > 0:
        print(f"\n=== Phase 1: Train classifier only for {head_epochs} epochs ===")
        print("⚠️  Phase 1 不使用 EMA（分类头从零开始训练）")
        
        set_trainable(model.backbone.features, False)
        set_trainable(model.backbone.classifier, True)
        optimizer = build_optimizer(model, lr_backbone=0.0, lr_head=lr_head)
        
        best_acc = 0.0
        model_save_dir = os.path.join(parent_dir, "model")
        os.makedirs(model_save_dir, exist_ok=True)
        phase1_model_path = os.path.join(model_save_dir, "phase1_best.pth")
        
        for epoch in range(head_epochs):
            print(f"\n[Phase1] Epoch {epoch+1}/{head_epochs}")
            # ✅ Phase 1 不使用 EMA（传入 ema=None）
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, 
                                                   scaler, use_amp, aug_cfg=aug_cfg_phase1, ema=None, amp_dtype=amp_dtype)
            # ✅ Phase 1 直接验证主模型（不用 EMA）
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp, 
                                        use_tta=False, amp_dtype=amp_dtype)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_model(model, phase1_model_path)
                print("✅ Saved new best model (Phase1)")
                
        # Phase 2 从 Phase 1 最佳权重开始
        print(f"\nLoading Phase1 best model (Val Acc: {best_acc:.4f})")
        model.load_state_dict(torch.load(phase1_model_path, weights_only=True))

    # ✅ Phase 2 开始时才初始化 EMA
    print("\n=== Phase 2: Fine-tune full network ===")
    
    if config.get("use_ema", False):
        print("✅ 初始化 EMA（Phase 2）")
        ema = ModelEMA(model, decay=float(config.get("ema_decay", 0.9998)), device=device)
    else:
        ema = None
    
    set_trainable(model, True)
    
    # 余弦退火 + Warmup
    optimizer = build_optimizer(model, lr_backbone=lr_backbone, lr_head=lr_head)
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"]-3, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[3])

    best_acc = 0.0
    patience_counter = 0
    max_patience = 7
    best_model_path = os.path.join(model_save_dir, "best_model.pth")

    for epoch in range(config["epochs"]):
        print(f"\n[Phase2] Epoch {epoch+1}/{config['epochs']} LR: {optimizer.param_groups[0]['lr']:.6f}")
        # ✅ Phase 2 使用 EMA
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, 
                                               scaler, use_amp, aug_cfg=aug_cfg_phase2, ema=ema, amp_dtype=amp_dtype)
        eval_model = ema.ema if ema is not None else model
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device, use_amp, 
                                    use_tta=False, amp_dtype=amp_dtype)  # ✅ Phase 2 也先禁用 TTA
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_model(eval_model, best_model_path)
            print("✅ Saved new best model!")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter} epochs")

        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break

    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.4f}")