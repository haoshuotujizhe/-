import json
import os
import pandas as pd
from collections import Counter
import sys

# ✅ 修复：获取正确的项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # code/ 目录
parent_dir = os.path.dirname(script_dir)  # submission_test/ 目录

# 加载配置
config_path = os.path.join(parent_dir, "model", "config.json")

print(f"🔍 脚本目录: {script_dir}")
print(f"🔍 项目根目录: {parent_dir}")
print(f"🔍 配置文件路径: {config_path}")
print(f"🔍 配置文件存在: {os.path.exists(config_path)}\n")

if not os.path.exists(config_path):
    print(f"❌ 错误：找不到配置文件 {config_path}")
    print(f"   请检查文件是否存在")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

def resolve_path(path):
    """将配置中的相对路径转换为绝对路径"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(parent_dir, path))

train_dir = resolve_path(config["train_dir"])
val_dir = resolve_path(config["val_dir"])
train_label_csv = resolve_path(config["train_label_csv"])

print("=" * 60)
print("📂 路径检查")
print("=" * 60)
print(f"训练集目录: {train_dir}")
print(f"验证集目录: {val_dir}")
print(f"标签文件: {train_label_csv}")
print(f"训练集存在: {os.path.exists(train_dir)}")
print(f"验证集存在: {os.path.exists(val_dir)}")
print(f"标签文件存在: {os.path.exists(train_label_csv)}")

# 读取 CSV
df = pd.read_csv(train_label_csv)
print(f"\n✅ CSV 文件读取成功，共 {len(df)} 条记录")
print(f"列名: {df.columns.tolist()}")

# 检查训练集图片
train_files = set(os.listdir(train_dir)) if os.path.exists(train_dir) else set()
val_files = set(os.listdir(val_dir)) if os.path.exists(val_dir) else set()

train_in_csv = df[df["filename"].isin(train_files)]
val_in_csv = df[df["filename"].isin(val_files)]

print("\n" + "=" * 60)
print("📊 数据集统计")
print("=" * 60)
print(f"训练集实际图片数: {len(train_files)}")
print(f"验证集实际图片数: {len(val_files)}")
print(f"CSV 中匹配到的训练集: {len(train_in_csv)} 张")
print(f"CSV 中匹配到的验证集: {len(val_in_csv)} 张")

# 检查类别分布
train_categories = train_in_csv["category_id"].astype(int).tolist()
val_categories = val_in_csv["category_id"].astype(int).tolist()

train_unique = set(train_categories)
val_unique = set(val_categories)

print("\n" + "=" * 60)
print("🏷️  类别分析")
print("=" * 60)
print(f"训练集类别数: {len(train_unique)}")
print(f"验证集类别数: {len(val_unique)}")
print(f"验证集独有类别数: {len(val_unique - train_unique)}")
print(f"训练集独有类别数: {len(train_unique - val_unique)}")

if val_unique - train_unique:
    print(f"\n❌ 危险！验证集包含训练集没有的类别:")
    print(f"   {sorted(val_unique - train_unique)[:10]}... (显示前10个)")

if train_unique - val_unique:
    print(f"\n⚠️  警告！训练集有 {len(train_unique - val_unique)} 个类别在验证集中不存在")

# 检查类别分布
print("\n" + "=" * 60)
print("📈 类别分布对比（前 10 个类别）")
print("=" * 60)
train_counts = Counter(train_categories)
val_counts = Counter(val_categories)

print(f"{'类别ID':<10} {'训练集样本数':<15} {'验证集样本数':<15}")
print("-" * 40)
for cat_id in sorted(train_unique)[:10]:
    train_count = train_counts.get(cat_id, 0)
    val_count = val_counts.get(cat_id, 0)
    print(f"{cat_id:<10} {train_count:<15} {val_count:<15}")

# 检查标签映射
print("\n" + "=" * 60)
print("🔢 标签映射检查")
print("=" * 60)
unique_train_sorted = sorted(train_unique)
label_mapping = {cat_id: idx for idx, cat_id in enumerate(unique_train_sorted)}
print(f"训练集标签映射（前 10 个）:")
for i, (cat_id, idx) in enumerate(list(label_mapping.items())[:10]):
    print(f"  category_id {cat_id} → 整数标签 {idx}")

# 检查验证集标签是否在映射中
unmapped_val = [cat for cat in val_unique if cat not in label_mapping]
if unmapped_val:
    print(f"\n❌ 严重错误！验证集有 {len(unmapped_val)} 个类别无法映射:")
    print(f"   {sorted(unmapped_val)[:10]}... (显示前10个)")
else:
    print(f"\n✅ 验证集所有类别都在训练集标签映射中")

# 检查图片是否可读
print("\n" + "=" * 60)
print("🖼️  图片完整性检查（随机抽样 10 张）")
print("=" * 60)
from PIL import Image
import random

sample_train = random.sample(list(train_in_csv["filename"]), min(10, len(train_in_csv)))
sample_val = random.sample(list(val_in_csv["filename"]), min(10, len(val_in_csv)))

def check_images(files, img_dir, label):
    print(f"\n{label}:")
    for fname in files:
        path = os.path.join(img_dir, fname)
        try:
            img = Image.open(path)
            img.verify()
            print(f"  ✅ {fname}: {img.size}")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

check_images(sample_train, train_dir, "训练集")
check_images(sample_val, val_dir, "验证集")

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)
