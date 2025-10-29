# code/utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import random
import numpy as np

from torch.utils.data import WeightedRandomSampler
from collections import Counter

class CSVBaseDataset(Dataset):
    """
    通用CSV数据集（支持训练集和验证集）
    从总CSV中筛选出指定图片目录下的图片，并使用统一的标签映射
    """
    def __init__(self, img_dir, total_csv, transform=None, label_mapping=None):
        """
        Args:
            img_dir: 图片目录（如 train_dir 或 val_dir）
            total_csv: 总的CSV标签文件路径（包含所有图片的标签）
            transform: 图像预处理
            label_mapping: 外部传入的标签映射（cat_id → 整数），训练集生成后传给验证集
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # 读取CSV
        self.df = pd.read_csv(total_csv)
        
        # 获取当前目录的文件列表
        files_in_dir = set(os.listdir(img_dir))
        
        # 只保留目录中实际存在的文件
        self.df = self.df[self.df["filename"].isin(files_in_dir)].reset_index(drop=True)
        
        self.filenames = self.df["filename"].tolist()
        self.category_ids = self.df["category_id"].astype(int).tolist()
        
        # ✅ 修复：建立标签映射
        if label_mapping is None:
            # 训练集：建立新的标签映射
            unique_categories = sorted(set(self.category_ids))
            self.label_mapping = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        else:
            # 验证集：使用训练集的标签映射
            self.label_mapping = label_mapping
        
        # ✅ 过滤掉验证集中无法映射的类别
        valid_indices = []
        valid_filenames = []
        valid_category_ids = []
        valid_integer_labels = []
        
        skipped_count = 0
        for i, cat_id in enumerate(self.category_ids):
            if cat_id in self.label_mapping:
                valid_indices.append(i)
                valid_filenames.append(self.filenames[i])
                valid_category_ids.append(cat_id)
                valid_integer_labels.append(self.label_mapping[cat_id])
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"⚠️  跳过 {skipped_count} 个无法映射的样本（验证集包含训练集中不存在的类别）")
        
        self.filenames = valid_filenames
        self.category_ids = valid_category_ids
        self.integer_labels = valid_integer_labels

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        label = self.integer_labels[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
            
def _build_weighted_sampler(integer_labels):
    #根据类别频次构造 WeightedRandomSampler（长尾数据更友好）
    counts = Counter(integer_labels)
        # 类别权重 = 1 / 频次
    class_weight = {c: 1.0 / cnt for c, cnt in counts.items()}
    sample_weights = [class_weight[y] for y in integer_labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                 num_samples=len(sample_weights),
                                 replacement=True)
        
def get_dataloaders(train_dir, train_label_csv, val_dir, config, use_strong_aug=False):
    mean, std = config["mean"], config["std"]
    input_size = tuple(config["input_size"])

    if use_strong_aug:
        # ✅ Phase 2 强增强（目标 99%）
        train_tf = transforms.Compose([
            # 更激进的裁剪（0.7-1.0，原来是 0.85-1.0）
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 新增垂直翻转
            transforms.RandomRotation(15),  # 增加旋转角度（原来 10°）
            # 更强的颜色抖动
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 新增平移
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # 更强的 RandomErasing（原来 0.15 → 0.25）
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
    else:
        # Phase 1 轻量增强
        train_tf = transforms.Compose([
            transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    val_tf = transforms.Compose([
        transforms.Resize((int(input_size[0] * 1.05), int(input_size[1] * 1.05))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ✅ 先加载训练集，建立标签映射
    train_set = CSVBaseDataset(img_dir=train_dir, total_csv=train_label_csv, transform=train_tf)
    
    print(f"✅ 训练集加载完成:")
    print(f"   样本数: {len(train_set)}")
    print(f"   类别数: {len(train_set.label_mapping)}")
    print(f"   类别范围: {min(train_set.label_mapping.keys())} ~ {max(train_set.label_mapping.keys())}")
    print(f"   映射后标签范围: {min(train_set.integer_labels)} ~ {max(train_set.integer_labels)}")
    
    # ✅ 验证集使用训练集的标签映射
    val_set = CSVBaseDataset(img_dir=val_dir, total_csv=train_label_csv, transform=val_tf, 
                             label_mapping=train_set.label_mapping)
    
    print(f"✅ 验证集加载完成:")
    print(f"   样本数: {len(val_set)}")
    print(f"   映射后标签范围: {min(val_set.integer_labels)} ~ {max(val_set.integer_labels)}")
    
    # ✅ 检查标签映射是否一致
    if set(val_set.integer_labels) - set(train_set.integer_labels):
        print(f"⚠️  警告：验证集有训练集没有的标签！")
    else:
        print(f"✅ 验证集标签全部在训练集中")
    print()

    # 加权采样
    if config.get("use_weighted_sampler", False):
        sampler = _build_weighted_sampler(train_set.integer_labels)
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], 
                                 sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], 
                                 shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=config["batch_size"], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 建议：确保可复现或更快
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path):
    torch.save(model.state_dict(), path)
