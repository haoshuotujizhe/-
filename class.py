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
    
    
    df=pd.read_csv(csv_path)    # 读取csv文件
    length = len(df)
    index = 0    # 循环遍历csv每一行（除第一行），对应的索引
    last_cat = 0    # 上一种类别的编号
    cat_num = 0     # 同种类别的数据数量

    for cat_id in df['category_id']:

        # 第一次初始化last_cat
        if index==0 :
            last_cat=cat_id    # 别忘加同类别计数增长和index增长

        # 与上一个类别不同（类别更替了）或是最后一行，则更新（移动上一个类别图片）
        elif cat_id!=last_cat or index==length-1 :
            if index==length-1 :
                cat_num+=1
                index+=1
            
            # 该类别验证集和测试集分别的数量
            move_val_num=int(cat_num*vt_weight_list[0])
            move_test_num=int(cat_num*vt_weight_list[1])
            
            # 判断是否超出该类别总数据量
            if cat_num<=move_val_num+move_test_num :
                print(f'beyond border error in line {index}.')
                break
            
            # 截取要移动的图片集
            move_imgs=df.iloc[index-move_test_num-move_val_num:index]

            temp_i=0    # 临时索引，用于判断某次处理的图片该移动到哪个位置，先val再test
            for img_name in move_imgs['filename'] :
                temp_i+=1
                old_path = os.path.join(original_train_dir, img_name)
                # 移动到val集
                if temp_i<=move_val_num :
                    new_path = os.path.join(val_set_dir, img_name)
                # 移动到test集
                else :
                    new_path = os.path.join(test_set_dir, img_name)
                # 移动
                if not os.path.exists(old_path):
                    print(f'[SKIP] 不存在: {old_path}')
                    continue
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                os.rename(old_path, new_path)

            # 类别更替则计数置零
            last_cat=cat_id
            cat_num=0

        cat_num+=1    # 其他情况该类别数据量加一
        index+=1

    return


if __name__ == "__main__":
    with open("model/config.json", "r") as f:
        config = json.load(f)
    dataset_classified(config)