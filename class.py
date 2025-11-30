import os
import random
import shutil
import json
import numpy as np
import pandas as pd
from collections import Counter


def analyze_distribution(df):
    """分析类别分布"""
    category_counts = df['category_id'].value_counts()
    
    print(f"\n{'='*60}")
    print("📊 类别分布分析")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")
    print(f"类别数: {len(category_counts)}")
    print(f"样本数范围: {category_counts.min()} ~ {category_counts.max()}")
    print(f"平均样本数: {category_counts.mean():.1f}")
    print(f"中位数: {category_counts.median():.1f}")
    
    # 长尾统计
    bins = [0, 20, 50, 100, 200, float('inf')]
    labels = ['≤20', '21-50', '51-100', '101-200', '>200']
    for i in range(len(bins) - 1):
        count = ((category_counts >= bins[i]) & (category_counts < bins[i+1])).sum()
        print(f"   {labels[i]} 样本: {count} 个类别")
    
    return category_counts


def stratified_split(df, val_ratio=0.15, min_val_per_class=2, seed=42):
    """
    分层划分数据集
    - 每个类别按比例划分
    - 小样本类别保证至少有验证样本
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_indices = []
    val_indices = []
    
    grouped = df.groupby('category_id')
    
    for cat_id, group in grouped:
        indices = group.index.tolist()
        n = len(indices)
        random.shuffle(indices)
        
        # 验证集数量（至少 min_val_per_class 个）
        n_val = max(min_val_per_class, int(n * val_ratio))
        
        # 确保训练集至少有 2 个样本
        if n - n_val < 2:
            n_val = max(1, n - 2)
        
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    
    return train_df, val_df


def dataset_classified(config):
    """将原始数据集划分为训练集和验证集"""
    
    original_dir = config["original_train_dir"]
    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    csv_path = config["train_label_csv"]
    val_ratio = config.get("val_ratio", 0.15)
    
    print(f"\n{'='*60}")
    print("🚀 开始数据集划分")
    print(f"{'='*60}")
    print(f"原始目录: {original_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    print(f"验证集比例: {val_ratio}")
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"\nCSV 总记录数: {len(df)}")
    
    # 检查原始目录
    if not os.path.exists(original_dir):
        print(f"❌ 原始目录不存在: {original_dir}")
        return None, None
    
    existing_files = set(os.listdir(original_dir))
    print(f"原始目录文件数: {len(existing_files)}")
    
    # 过滤存在的文件
    df = df[df['filename'].isin(existing_files)].reset_index(drop=True)
    print(f"匹配到的记录数: {len(df)}")
    
    if len(df) == 0:
        print("❌ 没有匹配到任何文件！")
        return None, None
    
    # 分析分布
    analyze_distribution(df)
    
    # 分层划分
    train_df, val_df = stratified_split(df, val_ratio=val_ratio, min_val_per_class=2, seed=42)
    
    print(f"\n{'='*60}")
    print("📂 划分结果")
    print(f"{'='*60}")
    print(f"训练集: {len(train_df)} 张 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 张 ({len(val_df)/len(df)*100:.1f}%)")
    
    # 检查类别覆盖
    train_cats = set(train_df['category_id'].unique())
    val_cats = set(val_df['category_id'].unique())
    
    print(f"\n训练集类别数: {len(train_cats)}")
    print(f"验证集类别数: {len(val_cats)}")
    
    missing = train_cats - val_cats
    if missing:
        print(f"⚠️ 验证集缺少 {len(missing)} 个类别")
    else:
        print("✅ 验证集包含所有训练集类别")
    
    # 创建目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 移动文件
    print(f"\n📦 移动文件中...")
    
    moved_train = 0
    moved_val = 0
    
    for _, row in train_df.iterrows():
        src = os.path.join(original_dir, row['filename'])
        dst = os.path.join(train_dir, row['filename'])
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_train += 1
    
    for _, row in val_df.iterrows():
        src = os.path.join(original_dir, row['filename'])
        dst = os.path.join(val_dir, row['filename'])
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_val += 1
    
    print(f"\n✅ 移动完成:")
    print(f"   训练集: {moved_train} 张")
    print(f"   验证集: {moved_val} 张")
    
    # 保存划分信息
    split_info = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "num_classes": len(train_cats),
        "val_ratio": val_ratio
    }
    
    info_path = os.path.join(os.path.dirname(csv_path), "split_info.json")
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n✅ 划分信息已保存: {info_path}")
    
    print(f"\n{'='*60}")
    print("🎉 数据集划分完成！")
    print(f"{'='*60}")
    
    return train_df, val_df


if __name__ == "__main__":
    # 加载配置
    config_path = "model/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 解析路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        if path.startswith("../"):
            path = path[3:]
        return os.path.join(project_root, path)
    
    config["original_train_dir"] = resolve_path(config.get("original_train_dir", "submission/datasets/original_train"))
    config["train_dir"] = resolve_path(config["train_dir"])
    config["val_dir"] = resolve_path(config["val_dir"])
    config["train_label_csv"] = resolve_path(config["train_label_csv"])
    
    print("路径配置:")
    print(f"  原始目录: {config['original_train_dir']}")
    print(f"  训练集: {config['train_dir']}")
    print(f"  验证集: {config['val_dir']}")
    print(f"  CSV: {config['train_label_csv']}")
    
    # 检查是否已划分
    if os.path.exists(config["train_dir"]) and len(os.listdir(config["train_dir"])) > 0:
        print("\n⚠️ 训练集目录已存在，跳过划分")
        print(f"   训练集: {len(os.listdir(config['train_dir']))} 张")
        print(f"   验证集: {len(os.listdir(config['val_dir']))} 张")
    else:
        dataset_classified(config)