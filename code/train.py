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
        # ✅ 同步参数 + BN buffers（避免 EMA 验证时统计量过期）
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if not torch.is_floating_point(esd[k]):
                # int buffer（num_batches_tracked 等）直接覆盖
                esd[k].copy_(msd[k])
            else:
                if ("running_mean" in k) or ("running_var" in k):
                    # BN running stats 直接同步，不做滑动平均
                    esd[k].copy_(msd[k])
                else:
                    # 其余可训练浮点参数做 EMA
                    esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)

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

def build_optimizer(model, lr_backbone, lr_head, weight_decay=5e-4):  # ✅ 从 1e-4 提升到 5e-4
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)  # ✅ 从 0.05 降到 0.02
    
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
    print(f"   Label Smoothing: 0.02")  # ✅ 更新显示

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
    print("🔄 重新加载训练数据（使用强增强）...")
    
    # ✅ 重新加载数据（使用强增强）
    train_loader, _ = get_dataloaders(
        train_dir=config["train_dir"],
        train_label_csv=config["train_label_csv"],
        val_dir=config["val_dir"],
        config=config,
        use_strong_aug=True  # ✅ Phase 2 使用强增强
    )
    
    if config.get("use_ema", False):
        print("✅ 初始化 EMA（Phase 2）")
        ema = ModelEMA(model, decay=float(config.get("ema_decay", 0.9998)), device=device)
    else:
        ema = None
    
    set_trainable(model, True)
    
    # ✅ 使用 Warmup + CosineAnnealingWarmRestarts（更稳定，周期性重启）
    optimizer = build_optimizer(model, lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=5e-4)  # ✅ 增加权重衰减
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    
    # Warmup 3 个 epoch
    def warmup_lambda(epoch):
        return (epoch + 1) / 3 if epoch < 3 else 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # ✅ 不重启的余弦退火，整个 Phase2 平滑下降
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=1e-6)

    best_acc = 0.0
    patience_counter = 0
    max_patience = 10       # 放宽早停（因为周期性重启）
    min_epochs = 25         # 最小训练轮数
    min_delta = 1e-4
    best_model_path = os.path.join(model_save_dir, "best_model.pth")

    for epoch in range(config["epochs"]):
        # ✅ 前 3 个 epoch 使用 warmup，之后使用 cosine
        if epoch < 3:
            current_lr = optimizer.param_groups[0]['lr'] * warmup_lambda(epoch)
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Phase2] Epoch {epoch+1}/{config['epochs']} LR: {current_lr:.6f}")
        
        # ✅ 动态减弱增强（第 15 个 epoch 后）
        cur_aug = dict(aug_cfg_phase2)
        if epoch >= 12:
            cur_aug.update({"use_cutmix": False, "mixup_alpha": 0.05})
            if epoch == 12:
                print("   📉 减弱增强：已关闭 CutMix，Mixup alpha=0.05")
        if epoch >= 18:
            cur_aug.update({"use_mixup": False, "mixup_alpha": 0.0, "use_cutmix": False})
            if epoch == 18:
                print("   📴 彻底关闭 Mixup/CutMix 以收敛")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, aug_cfg=cur_aug, ema=ema, amp_dtype=amp_dtype
        )
        eval_model = ema.ema if ema is not None else model
        if ema is not None:
            # ✅ 再次确保 BN buffers 与当前模型一致
            for (n, b_ema) in eval_model.named_buffers():
                b_model = dict(model.named_buffers())[n]
                b_ema.copy_(b_model)
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device, use_amp, 
                                    use_tta=False, amp_dtype=amp_dtype)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # ✅ 学习率调度
        if epoch < 3:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            patience_counter = 0
            save_model(eval_model, best_model_path)
            print("✅ Saved new best model!")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter} epochs")

        # ✅ 至少训练 min_epochs 后才早停
        if (epoch + 1) >= min_epochs and patience_counter >= max_patience:
            print("Early stopping triggered")
            break

    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.4f}")

    # ✅ Phase 3: 高分辨率微调（可显著抬最后 0.3~0.8%）
    if bool(config.get("final_finetune", True)):
        print("\n=== Phase 3: High-res fine-tune ===")
        cfg_p3 = dict(config)
        cfg_p3["input_size"] = list(config.get("final_input_size", [600, 600]))
        # 关强增强，只保留轻增强或 CenterCrop（在 utils 里按 use_strong_aug=False）
        train_loader_p3, _ = get_dataloaders(
            train_dir=cfg_p3["train_dir"],
            train_label_csv=cfg_p3["train_label_csv"],
            val_dir=cfg_p3["val_dir"],
            config=cfg_p3,
            use_strong_aug=False
        )

        # 小学习率微调（全部参数）
        lr_mult = float(config.get("final_lr_mult", 0.2))
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"] * lr_mult, 1e-6)

        # 关闭 Mixup/CutMix
        aug_p3 = {"use_mixup": False, "mixup_alpha": 0.0, "use_cutmix": False, "cutmix_alpha": 0.0}

        final_epochs = int(config.get("final_epochs", 3))
        for e in range(final_epochs):
            print(f"\n[Phase3] Epoch {e+1}/{final_epochs}")
            trl, tra = train_one_epoch(model, train_loader_p3, criterion, optimizer, device,
                                       scaler, use_amp, aug_cfg=aug_p3, ema=ema, amp_dtype=amp_dtype)
            eval_model = ema.ema if ema is not None else model
            if ema is not None:
                for (n, b_ema) in eval_model.named_buffers():
                    b_model = dict(model.named_buffers())[n]
                    b_ema.copy_(b_model)
            vll, vla = validate(eval_model, val_loader, criterion, device, use_amp, use_tta=True, amp_dtype=amp_dtype)
            print(f"Train Loss: {trl:.4f}, Acc: {tra:.4f} | Val Loss: {vll:.4f}, Acc: {vla:.4f}")
            if vla > best_acc:
                best_acc = vla
                save_model(eval_model, best_model_path)
                print("✅ Saved new best model (Phase3)")