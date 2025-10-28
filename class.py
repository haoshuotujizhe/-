import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import random
import numpy as np
import json

def dataset_classified(config) :
 
    """
        Args:
            _dir: dir用相对目录路径，如 ./datasets/val
            csv_path: 总的CSV标签文件路径（包含所有图片的标签）
            vt_weight_list: 就是val和test权重列表，比如分别占一个类别的[0.15,0.1]
    """
       
    vt_weight_list=config["vt_weight_list"] 
    original_train_dir =config["train_dir"]
    val_set_dir =config["val_dir"]
    test_set_dir =config["test_dir"]
    csv_path =config["train_label_csv"]
    
    # 设置随机种子以保证可复现性
    random.seed(42)
    np.random.seed(42)
    
    df=pd.read_csv(csv_path)    # 读取csv文件
    length = len(df)
    
    # 按类别分组处理
    grouped = df.groupby('category_id')
    
    for cat_id, group in grouped:
        cat_num = len(group)
        
        # 该类别验证集和测试集分别的数量
        move_val_num = int(cat_num * vt_weight_list[0])
        move_test_num = int(cat_num * vt_weight_list[1])
        
        # 判断是否超出该类别总数据量
        if cat_num <= move_val_num + move_test_num:
            print(f'⚠️ 类别 {cat_id} 样本不足: 总共{cat_num}张, 需要val={move_val_num}+test={move_test_num}')
            continue
        
        # ✅ 关键修改：随机打乱后再选择验证集/测试集
        indices = list(range(len(group)))
        random.shuffle(indices)
        
        val_indices = indices[:move_val_num]
        test_indices = indices[move_val_num:move_val_num + move_test_num]
        
        # 处理验证集图片
        for idx in val_indices:
            img_name = group.iloc[idx]['filename']
            old_path = os.path.join(original_train_dir, img_name)
            new_path = os.path.join(val_set_dir, img_name)
            
            if not os.path.exists(old_path):
                print(f'[SKIP] 不存在: {old_path}')
                continue
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)
        
        # 处理测试集图片
        for idx in test_indices:
            img_name = group.iloc[idx]['filename']
            old_path = os.path.join(original_train_dir, img_name)
            new_path = os.path.join(test_set_dir, img_name)
            
            if not os.path.exists(old_path):
                print(f'[SKIP] 不存在: {old_path}')
                continue
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)
        
        print(f'✅ 类别 {cat_id}: 训练集{cat_num - move_val_num - move_test_num}, 验证集{move_val_num}, 测试集{move_test_num}')

    return


if __name__ == "__main__":
    with open("model/config.json", "r") as f:
        config = json.load(f)
    dataset_classified(config)