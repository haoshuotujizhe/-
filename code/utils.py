import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image, ImageFile
import os
import random
import numpy as np
from collections import Counter

# ✅ 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ 硬编码 152 个类别（与 predict.py 完全一致）
CATEGORY_IDS = [
    164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291,
    1734, 1743, 1747, 1749, 1750, 1751, 1759, 1765, 1770, 1772, 1774, 1776,
    1777, 1780, 1784, 1785, 1786, 1789, 1796, 1797, 1801, 1805, 1806, 1808,
    1818, 1827, 1833
]

# 固定映射：category_id → index
CATEGORY_TO_INDEX = {cat_id: idx for idx, cat_id in enumerate(CATEGORY_IDS)}


def verify_image(img_path):
    """验证图片是否可以正常打开"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        # verify() 后需要重新打开
        with Image.open(img_path) as img:
            img.load()
        return True
    except Exception:
        return False


def compute_class_weights(class_counts, num_classes, smoothing=0.15):
    """计算类别权重（Effective Number of Samples 方法）"""
    weights = torch.ones(num_classes, dtype=torch.float32)
    
    if not class_counts:
        return weights
    
    counts = []
    for i in range(num_classes):
        counts.append(class_counts.get(i, 1))
    
    counts = np.array(counts, dtype=np.float32)
    
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    weights_np = (1.0 - beta) / (effective_num + 1e-8)
    weights_np = weights_np / weights_np.sum() * num_classes
    weights_np = smoothing + (1 - smoothing) * weights_np
    
    return torch.from_numpy(weights_np).float()


class CSVBaseDataset(Dataset):
    """使用硬编码 CATEGORY_IDS 的数据集，带图片验证"""
    
    def __init__(self, img_dir, total_csv, transform=None, label_mapping=None, verify_images=True):
        self.img_dir = img_dir
        self.transform = transform
        
        self.df = pd.read_csv(total_csv)
        
        files_in_dir = set(os.listdir(img_dir))
        self.df = self.df[self.df["filename"].isin(files_in_dir)].reset_index(drop=True)
        
        all_filenames = self.df["filename"].tolist()
        all_category_ids = self.df["category_id"].astype(int).tolist()
        
        self.label_mapping = CATEGORY_TO_INDEX
        
        # 过滤有效样本
        self.filenames = []
        self.category_ids = []
        self.integer_labels = []
        skipped_cat = 0
        skipped_img = 0
        
        for i, cat_id in enumerate(all_category_ids):
            if cat_id not in self.label_mapping:
                skipped_cat += 1
                continue
            
            img_path = os.path.join(img_dir, all_filenames[i])
            
            # ✅ 验证图片（可选，首次运行建议开启）
            if verify_images:
                if not verify_image(img_path):
                    skipped_img += 1
                    continue
            
            self.filenames.append(all_filenames[i])
            self.category_ids.append(cat_id)
            self.integer_labels.append(self.label_mapping[cat_id])
        
        if skipped_cat > 0:
            print(f"⚠️ 跳过 {skipped_cat} 个不在 CATEGORY_IDS 中的样本")
        if skipped_img > 0:
            print(f"⚠️ 跳过 {skipped_img} 个损坏的图片")
        
        self.class_counts = Counter(self.integer_labels)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        label = self.integer_labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # ✅ 如果仍然出错，返回随机有效样本
            print(f"⚠️ 加载失败 {img_path}: {e}")
            new_idx = random.randint(0, len(self.filenames) - 1)
            return self.__getitem__(new_idx)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def _build_weighted_sampler(integer_labels, power=0.5):
    """加权采样器"""
    counts = Counter(integer_labels)
    class_weight = {c: 1.0 / (cnt ** power) for c, cnt in counts.items()}
    sample_weights = [class_weight[y] for y in integer_labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloaders(train_dir, train_label_csv, val_dir, config, use_strong_aug=False, progressive_strength=1.0):
    """返回 train_loader, val_loader, class_counts"""
    mean, std = config["mean"], config["std"]
    input_size = tuple(config["input_size"])

    if use_strong_aug:
        s = progressive_strength
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.65 + 0.15*(1-s), 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3 * s),
            transforms.RandomRotation(int(20 * s)),
            transforms.ColorJitter(
                brightness=0.35 * s,
                contrast=0.35 * s,
                saturation=0.35 * s,
                hue=0.1 * s
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1 * s, 0.1 * s)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25 * s, scale=(0.02, 0.2))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    val_tf = transforms.Compose([
        transforms.Resize((int(input_size[0] * 1.05), int(input_size[1] * 1.05))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ✅ 首次加载时验证图片
    print("🔍 验证训练集图片...")
    train_set = CSVBaseDataset(img_dir=train_dir, total_csv=train_label_csv, transform=train_tf, verify_images=True)
    
    print(f"✅ 训练集: {len(train_set)} 样本, {len(train_set.class_counts)} 类别")
    print(f"   标签范围: {min(train_set.integer_labels)} ~ {max(train_set.integer_labels)}")
    
    print("🔍 验证验证集图片...")
    val_set = CSVBaseDataset(img_dir=val_dir, total_csv=train_label_csv, transform=val_tf, verify_images=True)
    print(f"✅ 验证集: {len(val_set)} 样本")

    num_workers = int(config.get("num_workers", 8))
    persistent_workers = bool(config.get("persistent_workers", True)) and (num_workers > 0)
    prefetch_factor = int(config.get("prefetch_factor", 4))

    if config.get("use_weighted_sampler", False):
        sampler = _build_weighted_sampler(train_set.integer_labels, power=0.5)
        train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, drop_last=True
        )
        print("✅ 启用加权采样器")
    else:
        train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, drop_last=True
        )

    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )
    
    return train_loader, val_loader, train_set.class_counts


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)