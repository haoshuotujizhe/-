# code/utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import random
import numpy as np

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
        self.df = pd.read_csv(total_csv)  # 读取总CSV
        
        # 筛选出当前图片目录下存在的图片（只保留这些条目标签）
        self.df = self._filter_existing_images(self.df, img_dir)
        
        # 提取category_id（转为整数）
        self.category_ids = self.df["category_id"].astype(int).tolist()
        self.filenames = self.df["filename"].tolist()
        
        # 处理标签映射：训练集自己生成，验证集复用训练集的
        if label_mapping is None:
            # 训练集：生成category_id → 整数的映射（0开始连续）
            self.unique_category_ids = sorted(list(set(self.category_ids)))
            self.label_mapping = {cat_id: idx for idx, cat_id in enumerate(self.unique_category_ids)}
        else:
            # 验证集：复用训练集的映射（确保标签一致）
            self.label_mapping = label_mapping
        
        # 转换为模型需要的整数标签
        self.integer_labels = [self.label_mapping[cat_id] for cat_id in self.category_ids]

    def _filter_existing_images(self, df, img_dir):
        """筛选出在img_dir中实际存在的图片条目"""
        existing_filenames = []
        for idx, row in df.iterrows():
            img_path = os.path.join(img_dir, row["filename"])
            if os.path.exists(img_path):
                existing_filenames.append(row["filename"])
        # 只保留存在的图片
        return df[df["filename"].isin(existing_filenames)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        label = self.integer_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
def get_dataloaders(train_dir,train_label_csv,val_dir, config): #此处参数传入的是，训练用的图片的地址，带标签的csv文件的地址，验证用的图片的地址，模型参数json文件的地址
    mean, std = config["mean"], config["std"]
    input_size = tuple(config["input_size"])

    train_tf = transforms.Compose([
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # 保留主体为主
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # 新增垂直翻转
    transforms.RandomRotation(20),  # 新增旋转
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # 轻微颜色扰动
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),  # 自动增强策略
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))  # 不宜太强
])
    resize_side = int(max(input_size) * 1.15)
    val_tf = transforms.Compose([
    transforms.Resize(resize_side),   # 稍微放大一点
    transforms.CenterCrop(input_size),           # 居中裁剪到模型输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


    train_set = CSVBaseDataset(img_dir=train_dir,  total_csv=train_label_csv,transform=train_tf)
    val_set = CSVBaseDataset(img_dir=val_dir,  total_csv=train_label_csv,transform=val_tf,label_mapping=train_set.label_mapping)

    num_workers = max(2, (os.cpu_count() or 4) // 2)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=pin,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=pin)
    
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
