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
    def __init__(self, model, decay=0.9999, device=None):
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
        # ✅ 同步参数 + BN buffers
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if not torch.is_floating_point(esd[k]):
                esd[k].copy_(msd[k])
            else:
                if ("running_mean" in k) or ("running_var" in k):
                    esd[k].copy_(msd[k])
                else:
                    esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)

def copy_model(model):
    import copy
    m = copy.deepcopy(model)
    return m

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, max_grad_norm=1.0, 
                    aug_cfg=None, ema=None, amp_dtype=torch.float16):
    model.train()
    # ✅ Phase 3 需要保持 BN 冻结
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
        total_correct += (preds == original_labels).sum().item()  
        total += labels.size(0)
    return total_loss / total, total_correct / total

def validate(model, dataloader, criterion, device,use_amp=False, use_tta=False, amp_dtype=torch.float16):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            if use_tta:
                # ✅ 花朵识别：只用水平翻转
                imgs_h = torch.flip(imgs, dims=[3])
                
                if AMP_HAS_DEVICE:
                    with autocast(device_type="cuda",dtype=amp_dtype,enabled=use_amp):
                        out1 = model(imgs)
                        out2 = model(imgs_h)
                        outputs = (out1 + out2) / 2.0
                        loss = criterion(outputs, labels)
                else:
                    with autocast(enabled=use_amp):
                        out1 = model(imgs)
                        out2 = model(imgs_h)
                        outputs = (out1 + out2) / 2.0
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

def build_optimizer(model, lr_backbone, lr_head, weight_decay=5e-4):
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
    config_path = os.path.join(os.path.dirname(__file__), "../model/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # ✅ 启用 TF32
    if bool(config.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 将配置中的相对路径转换为绝对路径
    parent_dir = os.path.dirname(os.path.dirname(__file__))
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
    amp_dtype = torch.float16
    
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
    
    # 模型
    model = build_model(config)
    # ✅ channels_last
    if bool(config.get("channels_last", True)):
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    
    # ✅ 更强的正则化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.015)  # 从 0.01 → 0.015
    
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
    print(f"   Label Smoothing: 0.01")

    model_save_dir = "../submission/models"
    os.makedirs(model_save_dir, exist_ok=True)
    scaler = GradScaler(enabled=use_amp)

    # Phase 1
    if head_epochs > 0:
        print(f"\n=== Phase 1: Train classifier only for {head_epochs} epochs ===")
        print("⚠️  Phase 1 不使用 EMA（分类头从零开始训练）")
        
        set_trainable(model.backbone.features, False)
        set_trainable(model.backbone.classifier, True)
        
        optimizer = build_optimizer(model, lr_backbone=0.0, lr_head=lr_head, weight_decay=2e-4)  # 从 1e-4 → 2e-4
        
        best_acc = 0.0
        phase1_model_path = os.path.join(model_save_dir, "phase1_best.pth")
        
        for epoch in range(head_epochs):
            print(f"\n[Phase1] Epoch {epoch+1}/{head_epochs}")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler, use_amp, aug_cfg=aug_cfg_phase1, ema=None, amp_dtype=amp_dtype
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp, 
                                        use_tta=False, amp_dtype=amp_dtype)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_model(model, phase1_model_path)
                print("✅ Saved new best model (Phase1)")
        
        print(f"\nLoading Phase1 best model (Val Acc: {best_acc:.4f})")
        if best_acc < 0.75:
            print("⚠️ Phase 1 准确率过低，考虑增加 head_epochs 或调整学习率")
        model.load_state_dict(torch.load(phase1_model_path, map_location=device))

    # Phase 2
    print("\n=== Phase 2: Fine-tune full network ===")
    print("🔄 重新加载训练数据（使用强增强）...")
    
    train_loader, _ = get_dataloaders(
        train_dir=config["train_dir"],
        train_label_csv=config["train_label_csv"],
        val_dir=config["val_dir"],
        config=config,
        use_strong_aug=True
    )
    
    if config.get("use_ema", False):
        print("✅ 初始化 EMA（Phase 2）")
        ema = ModelEMA(model, decay=float(config.get("ema_decay", 0.9999)), device=device)
    
    set_trainable(model, True)
    
    optimizer = build_optimizer(model, lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=5e-4)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    
    def warmup_lambda(epoch):
        return (epoch + 1) / 3 if epoch < 3 else 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # ✅ 单调余弦退火（不重启）
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=1e-8)

    best_acc = 0.0
    patience_counter = 0
    max_patience = 20  # 从 12 → 20
    min_epochs = 50    # 从 35 → 50（确保充分训练）
    min_delta = 1e-5   # 从 5e-5 → 1e-5（更敏感）
    best_model_path = os.path.join(model_save_dir, "best_model.pth")

    tta_val_every = int(config.get("tta_val_every", 5))
    # ✅ 数据完美，可以更早关闭增强
    turn_off_cutmix_epoch = 18  # epoch 18 关闭 CutMix
    turn_off_mixup_epoch = 30   # epoch 30 关闭 Mixup
    turn_off_all_aug_epoch = 999  # 永远保持基础增强
    
    for epoch in range(config["epochs"]):
        current_lr = optimizer.param_groups[0]['lr'] * warmup_lambda(epoch) if epoch < 3 else optimizer.param_groups[0]['lr']
        print(f"\n[Phase2] Epoch {epoch+1}/{config['epochs']} LR: {current_lr:.6f}")
        
        cur_aug = dict(aug_cfg_phase2)
        if epoch >= turn_off_cutmix_epoch:
            cur_aug.update({"use_cutmix": False, "mixup_alpha": 0.15})
            if epoch == turn_off_cutmix_epoch:
                print("   📉 关闭 CutMix，保持 Mixup alpha=0.15")
        
        if epoch >= turn_off_mixup_epoch:
            cur_aug.update({"use_cutmix": False, "mixup_alpha": 0.0, "use_mixup": False})
            if epoch == turn_off_mixup_epoch:
                print("   📴 关闭所有混合增强，开始最终收敛")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, aug_cfg=cur_aug, ema=ema, amp_dtype=amp_dtype
        )
        
        eval_model = ema.ema if ema is not None else model
        if ema is not None:
            for (n, b_ema) in eval_model.named_buffers():
                b_model = dict(model.named_buffers())[n]
                b_ema.copy_(b_model)
        
        use_tta_val = ((epoch + 1) % tta_val_every == 0) or ((epoch + 1) >= config["epochs"] - 5)
        if use_tta_val:
            print("   🔍 使用 TTA 验证...")
        
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device, use_amp, 
                                    use_tta=use_tta_val, amp_dtype=amp_dtype)
        
        # ✅ 显示训练和验证的差距
        gap = abs(train_acc - val_acc)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Gap: {gap:.4f}")
        
        (warmup_scheduler if epoch < 3 else cosine_scheduler).step()
        
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            patience_counter = 0
            save_model(eval_model, best_model_path)
            print("✅ Saved new best model!")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter}/{max_patience} epochs")
        
        if (epoch + 1) >= min_epochs and patience_counter >= max_patience:
            print("Early stopping triggered")
            break
    
    print(f"\n🎉 Final Best Val Accuracy: {best_acc:.4f}")