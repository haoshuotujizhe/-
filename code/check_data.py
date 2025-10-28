import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from utils import get_dataloaders, set_seed
import matplotlib.pyplot as plt

# 加载配置
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(parent_dir, "model", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# 解析路径
def resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(parent_dir, p))

config["train_dir"] = resolve_path(config["train_dir"])
config["val_dir"] = resolve_path(config["val_dir"])
config["train_label_csv"] = resolve_path(config["train_label_csv"])

set_seed(42)

# 加载数据
train_loader, val_loader = get_dataloaders(
    train_dir=config["train_dir"],
    train_label_csv=config["train_label_csv"],
    val_dir=config["val_dir"],
    config=config,
    use_strong_aug=False
)

print("=" * 60)
print("数据集诊断报告")
print("=" * 60)

# 训练集统计
print(f"\n训练集:")
print(f"  - 样本数量: {len(train_loader.dataset)}")
print(f"  - 批次数量: {len(train_loader)}")
print(f"  - 类别映射: {train_loader.dataset.label_mapping}")
print(f"  - 类别数量: {len(train_loader.dataset.label_mapping)}")

# 验证集统计
print(f"\n验证集:")
print(f"  - 样本数量: {len(val_loader.dataset)}")
print(f"  - 批次数量: {len(val_loader)}")
print(f"  - 使用训练集映射: {val_loader.dataset.label_mapping == train_loader.dataset.label_mapping}")

# 检查标签分布
from collections import Counter
train_label_dist = Counter(train_loader.dataset.integer_labels)
val_label_dist = Counter(val_loader.dataset.integer_labels)

print(f"\n训练集标签分布:")
print(f"  - 最多的类别: {train_label_dist.most_common(3)}")
print(f"  - 最少的类别: {train_label_dist.most_common()[-3:]}")

print(f"\n验证集标签分布:")
print(f"  - 最多的类别: {val_label_dist.most_common(3)}")
print(f"  - 最少的类别: {val_label_dist.most_common()[-3:]}")

# 检查样本
print(f"\n样本检查:")
imgs, labels = next(iter(train_loader))
print(f"  - 图像张量形状: {imgs.shape}")
print(f"  - 标签张量形状: {labels.shape}")
print(f"  - 图像值范围: [{imgs.min():.3f}, {imgs.max():.3f}]")
print(f"  - 批次中的标签: {labels.tolist()}")

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)
