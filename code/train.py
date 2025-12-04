import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import json
import shutil
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy
import sys

try:
    from torch.amp import autocast, GradScaler
    AMP_HAS_DEVICE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_HAS_DEVICE = False

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import build_model
from utils import get_dataloaders, save_model, set_seed, compute_class_weights, CATEGORY_IDS


def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images, targets, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha):
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
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, (targets, targets[index], lam)
    return images, (targets, None, 1.0)


class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        if device:
            self.ema = self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            if torch.is_floating_point(esd[k]):
                if "running" in k:
                    esd[k].copy_(msd[k])
                else:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1-self.decay)
            else:
                esd[k].copy_(msd[k])


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n = pred.size(-1)
        log_p = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_p)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -true_dist * log_p
        if self.weight is not None:
            loss = loss * self.weight[target].unsqueeze(1)
        
        return loss.sum(dim=-1).mean()


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=1e-5, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs  # 最少训练轮数
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score, epoch):
        if epoch < self.min_epochs:
            # 未达到最少轮数，只更新 best_score
            if self.best_score is None or score > self.best_score + self.min_delta:
                self.best_score = score
            return False
        
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self, keep_best=False):
        if not keep_best:
            self.best_score = None
        self.counter = 0
        self.should_stop = False


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, aug_cfg, ema, amp_dtype):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        if imgs.device.type == 'cuda':
            imgs = imgs.to(memory_format=torch.channels_last)
        
        orig_labels = labels.clone()
        imgs, (y_a, y_b, lam) = apply_mixup_cutmix(
            imgs, labels,
            aug_cfg.get("use_mixup", False), aug_cfg.get("mixup_alpha", 0),
            aug_cfg.get("use_cutmix", False), aug_cfg.get("cutmix_alpha", 0)
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if AMP_HAS_DEVICE else autocast(enabled=use_amp)
        with ctx:
            out = model(imgs)
            loss = criterion(out, y_a) if y_b is None else lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
        
        if use_amp and scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if ema:
            ema.update(model)
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        total_correct += (preds == orig_labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, total_correct / total


def validate(model, loader, criterion, device, use_amp, use_tta, amp_dtype):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            if imgs.device.type == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            
            ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if AMP_HAS_DEVICE else autocast(enabled=use_amp)
            
            with ctx:
                if use_tta:
                    out1 = model(imgs)
                    out2 = model(torch.flip(imgs, [3]))
                    out3 = model(torch.flip(imgs, [2]))
                    out = (out1 + out2 + out3) / 3
                else:
                    out = model(imgs)
                loss = criterion(out, labels)
            
            total_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, total_correct / total


def set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def build_optimizer(model, lr_backbone, lr_head, wd=5e-4):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "bias" in n or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    
    head_params = set(id(p) for p in model.backbone.classifier.parameters())
    
    groups = [
        {"params": [p for p in decay if id(p) not in head_params], "lr": lr_backbone, "weight_decay": wd},
        {"params": [p for p in no_decay if id(p) not in head_params], "lr": lr_backbone, "weight_decay": 0},
        {"params": [p for p in decay if id(p) in head_params], "lr": lr_head, "weight_decay": wd},
        {"params": [p for p in no_decay if id(p) in head_params], "lr": lr_head, "weight_decay": 0},
    ]
    return optim.AdamW(groups)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../model/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 一致性检查
    if config["num_classes"] != len(CATEGORY_IDS):
        raise ValueError(f"num_classes({config['num_classes']}) != len(CATEGORY_IDS)({len(CATEGORY_IDS)})")
    print(f"类别数检查: {config['num_classes']}")
    
    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    resolve = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(parent_dir, p))
    
    config["train_dir"] = resolve(config["train_dir"])
    config["val_dir"] = resolve(config["val_dir"])
    config["train_label_csv"] = resolve(config["train_label_csv"])
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    
    # 数据加载
    train_loader, val_loader, class_counts = get_dataloaders(
        config["train_dir"], config["train_label_csv"], config["val_dir"], config
    )
    
    # 模型
    model = build_model(config)
    model = model.to(device, memory_format=torch.channels_last) if config.get("channels_last") else model.to(device)
    
    # 类别权重
    class_weights = None
    if config.get("use_class_weight_loss"):
        class_weights = compute_class_weights(class_counts, config["num_classes"]).to(device)
        print(f"类别权重: {class_weights.min():.2f} ~ {class_weights.max():.2f}")
    
    # 损失函数
    smoothing = config.get("label_smoothing", 0.05)
    criterion = LabelSmoothingCE(smoothing=smoothing, weight=class_weights)
    
    model_save_dir = os.path.join(parent_dir, "submission/models")
    os.makedirs(model_save_dir, exist_ok=True)
    scaler = GradScaler(enabled=use_amp)
    
    head_epochs = config.get("head_epochs", 12)
    lr_backbone = config.get("lr_backbone", 0.000015)
    lr_head = config.get("lr_head", 0.00012)
    
    aug_off = {"use_mixup": False, "mixup_alpha": 0, "use_cutmix": False, "cutmix_alpha": 0}
    aug_on = {
        "use_mixup": config.get("use_mixup", True),
        "mixup_alpha": config.get("mixup_alpha", 0.2),
        "use_cutmix": config.get("use_cutmix", True),
        "cutmix_alpha": config.get("cutmix_alpha", 0.35)
    }
    
    # 早停配置
    p1_patience = config.get("p1_patience", 5)      # Phase 1 耐心
    p2_patience = config.get("p2_patience", 15)     # Phase 2 耐心
    p3_patience = config.get("p3_patience", 8)      # Phase 3 耐心
    
    # ========== Phase 1 ==========
    if head_epochs > 0:
        print(f"\n{'='*50}\n=== Phase 1: Head Only ({head_epochs} epochs, patience={p1_patience}) ===\n{'='*50}")
        
        set_trainable(model, False)
        set_trainable(model.backbone.classifier, True)
        
        opt = build_optimizer(model, 0, lr_head, 2e-4)
        best_acc = 0
        p1_path = os.path.join(model_save_dir, "phase1_best.pth")
        
        # Phase 1 早停
        early_stop_p1 = EarlyStopping(patience=p1_patience, min_delta=1e-4, min_epochs=3)
        
        for ep in range(head_epochs):
            print(f"\n[P1] Epoch {ep+1}/{head_epochs}")
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, opt, device, scaler, use_amp, aug_off, None, amp_dtype)
            v_loss, v_acc = validate(model, val_loader, criterion, device, use_amp, False, amp_dtype)
            print(f"Train: {t_loss:.4f}/{t_acc:.4f} | Val: {v_loss:.4f}/{v_acc:.4f}")
            
            if v_acc > best_acc:
                best_acc = v_acc
                save_model(model, p1_path)
                print("Saved P1 best")
            
            # 检查早停
            if early_stop_p1(v_acc, ep):
                print(f"P1 Early stop at epoch {ep+1}")
                break
        
        model.load_state_dict(torch.load(p1_path, map_location=device))
        print(f"P1 Best: {best_acc:.4f}")
    
    # ========== Phase 2 ==========
    print(f"\n{'='*50}\n=== Phase 2: Full Fine-tune ({config['epochs']} epochs, patience={p2_patience}) ===\n{'='*50}")
    
    train_loader, _, _ = get_dataloaders(
        config["train_dir"], config["train_label_csv"], config["val_dir"], config, use_strong_aug=True
    )
    
    set_trainable(model, True)
    ema = ModelEMA(model, config.get("ema_decay", 0.99995), device) if config.get("use_ema") else None
    
    opt = build_optimizer(model, lr_backbone, lr_head)
    
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    warmup_ep = config.get("warmup_epochs", 5)
    warmup_sched = LambdaLR(opt, lambda e: (e+1)/warmup_ep if e < warmup_ep else 1.0)
    cosine_sched = CosineAnnealingLR(opt, T_max=config["epochs"], eta_min=config.get("min_lr", 1e-7))
    
    best_acc = 0
    best_path = os.path.join(model_save_dir, "best_model.pth")
    
    # Phase 2 早停
    early_stop_p2 = EarlyStopping(patience=p2_patience, min_delta=1e-5, min_epochs=20)
    
    for ep in range(config["epochs"]):
        lr_now = opt.param_groups[0]['lr']
        print(f"\n[P2] Epoch {ep+1}/{config['epochs']} LR: {lr_now:.2e} (no_improve: {early_stop_p2.counter}/{p2_patience})")
        
        # 动态关闭增强
        cur_aug = dict(aug_on)
        if ep >= 20:
            cur_aug["use_cutmix"] = False
        if ep >= 35:
            cur_aug["use_mixup"] = False
        
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, opt, device, scaler, use_amp, cur_aug, ema, amp_dtype)
        
        eval_m = ema.ema if ema else model
        use_tta = (ep+1) % config.get("tta_val_every", 3) == 0 or ep >= config["epochs"] - 5
        v_loss, v_acc = validate(eval_m, val_loader, criterion, device, use_amp, use_tta, amp_dtype)
        
        print(f"Train: {t_loss:.4f}/{t_acc:.4f} | Val: {v_loss:.4f}/{v_acc:.4f} | Gap: {abs(t_acc-v_acc):.4f}")
        
        (warmup_sched if ep < warmup_ep else cosine_sched).step()
        
        if v_acc > best_acc + 1e-5:
            best_acc = v_acc
            save_model(eval_m, best_path)
            print("Saved best!")
        
        # 检查早停
        if early_stop_p2(v_acc, ep):
            print(f"⏹ P2 Early stop at epoch {ep+1}")
            break
    
    print(f"\nP2 Best: {best_acc:.4f}")
    
    # ========== Phase 3 ==========
    if config.get("final_finetune"):
        print(f"\n{'='*50}\n=== Phase 3: High-Res Fine-tune (patience={p3_patience}) ===\n{'='*50}")
        
        model.load_state_dict(torch.load(best_path, map_location=device))
        if ema:
            ema = ModelEMA(model, config.get("ema_decay", 0.99995), device)
        
        config["input_size"] = config.get("final_finetune_size", [576, 576])
        train_loader, val_loader, _ = get_dataloaders(
            config["train_dir"], config["train_label_csv"], config["val_dir"], config, use_strong_aug=False
        )
        
        opt = build_optimizer(model, lr_backbone * 0.1, lr_head * 0.1, 1e-4)
        p3_epochs = config.get("final_finetune_epochs", 25)
        sched = CosineAnnealingLR(opt, T_max=p3_epochs, eta_min=1e-8)
        
        criterion_p3 = nn.CrossEntropyLoss()
        p3_best = best_acc
        
        # Phase 3 早停
        early_stop_p3 = EarlyStopping(patience=p3_patience, min_delta=1e-5, min_epochs=5)
        
        for ep in range(p3_epochs):
            print(f"\n[P3] Epoch {ep+1}/{p3_epochs} (no_improve: {early_stop_p3.counter}/{p3_patience})")
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion_p3, opt, device, scaler, use_amp, aug_off, ema, amp_dtype)
            eval_m = ema.ema if ema else model
            v_loss, v_acc = validate(eval_m, val_loader, criterion_p3, device, use_amp, True, amp_dtype)
            print(f"Train: {t_loss:.4f}/{t_acc:.4f} | Val: {v_loss:.4f}/{v_acc:.4f}")
            sched.step()
            
            if v_acc > p3_best + 1e-5:
                p3_best = v_acc
                save_model(eval_m, best_path)
                print("Saved P3 best!")
            
            # 检查早停
            if early_stop_p3(v_acc, ep):
                print(f"P3 Early stop at epoch {ep+1}")
                break
        
        best_acc = p3_best
    
    # 复制到 model/
    final_path = os.path.join(parent_dir, "model", "best_model.pth")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    shutil.copy(best_path, final_path)
    
    print(f"\n{'='*50}")
    print(f"训练完成! Best: {best_acc:.4f}")
    print(f"模型: {final_path}")
    print(f"{'='*50}")