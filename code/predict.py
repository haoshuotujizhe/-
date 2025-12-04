import os
import argparse
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


# 硬编码 152 个类别
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


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_path, num_classes, device):
    from model import build_model
    model = build_model({"num_classes": num_classes, "use_pretrained": False})
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def preprocess_batch(image_paths, size, mean, std):
    """批量预处理图片 - 优化版本"""
    tf = transforms.Compose([
        transforms.Resize((int(size[0]*1.05), int(size[1]*1.05))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    images = []
    valid_paths = []  # 记录成功处理的路径
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(tf(img))
            valid_paths.append(path)
        except Exception as e:
            # print(f"跳过损坏图片 {path.name}: {e}")
            continue
    
    if images:  # 确保列表不为空
        return torch.stack(images), valid_paths
    else:
        return torch.empty(0), []  # 返回空张量

def predict_batch(model, imgs_batch, device, use_tta=True):
    """批量预测"""
    imgs_batch = imgs_batch.to(device)
    batch_size = imgs_batch.shape[0]
    with torch.no_grad():
        if use_tta:
            # 仅保留翻转，去掉缩放，与 train.py 保持逻辑更接近
            outs = [model(imgs_batch)]
            outs.append(model(torch.flip(imgs_batch, [3]))) # 水平
            outs.append(model(torch.flip(imgs_batch, [2]))) # 垂直
            # outs.append(model(torch.flip(imgs_batch, [2, 3]))) # 双翻转 (可选，train.py里没用这个)
            
            # 去掉这个缩放，因为它可能导致细节丢失
            # s = F.interpolate(...) 
            # outs.append(model(s))
            
            out = torch.mean(torch.stack(outs), dim=0)
        else:
            out = model(imgs_batch)
        
        probs = F.softmax(out, dim=1)
        confs, preds = probs.max(1)  # preds: [batch_size], confs: [batch_size]
    return preds.cpu().numpy(), confs.cpu().numpy()
def preprocess(path, size, mean, std):
    tf = transforms.Compose([
        transforms.Resize((int(size[0]*1.05), int(size[1]*1.05))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)

def predict(model, img, device, use_tta=True):
    img = img.to(device)
    with torch.no_grad():
        if use_tta:
            # 修改：与 predict_batch 保持完全一致 (原图 + 水平 + 垂直)
            outs = [model(img)]
            outs.append(model(torch.flip(img, [3]))) # 水平
            outs.append(model(torch.flip(img, [2]))) # 垂直
            
            # 删除双翻转和缩放，防止逻辑不一致
            # outs.append(model(torch.flip(img, [2, 3])))
            # s = F.interpolate(...)
            # outs.append(model(s))
            
            out = torch.mean(torch.stack(outs), dim=0)
        else:
            out = model(img)
        
        probs = F.softmax(out, dim=1)
        conf, pred = probs.max(1)
    return pred.item(), conf.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_img_dir", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "model" / "config.json")
    
    # 一致性检查
    if config["num_classes"] != len(CATEGORY_IDS):
        raise ValueError(f"num_classes mismatch: {config['num_classes']} vs {len(CATEGORY_IDS)}")
    
    model = load_model(root / "model" / "best_model.pth", config["num_classes"], device)
    model = model.to(dtype=dtype)
    
    img_dir = Path(args.test_img_dir)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    
    # 核心修复：优先使用 Phase 3 的高分辨率尺寸
    if config.get("final_finetune", False) and "final_finetune_size" in config:
        size = tuple(config["final_finetune_size"])
        # print(f"检测到 Phase 3 微调，使用高分辨率进行预测: {size}")
    else:
        size = tuple(config["input_size"])
        # print(f"ℹ使用标准分辨率进行预测: {size}")
    
    mean, std = config["mean"], config["std"]
    # 建议开启 TTA 以获得这最后的 0.5% 提升
    use_tta = config.get("use_tta", True) 
    batch_size = args.batch_size
    
    results = []
    processed_count = 0
    
    # print(f"开始处理 {len(imgs)} 张图片，批量大小: {batch_size}")
    
    for i in range(0, len(imgs), batch_size):
        batch_paths = imgs[i:i+batch_size]
        
        # 【修改】获取预处理结果和有效路径
        t_batch, valid_paths = preprocess_batch(batch_paths, size, mean, std)
        
        # 如果没有成功预处理的图片，跳过
        if len(t_batch) == 0:
            continue
            
        # 【修改】只在CUDA时转半精度
        if device.type == "cuda":
            t_batch = t_batch.half()
        else:
            t_batch = t_batch.float()
            
        try:
            preds, confs = predict_batch(model, t_batch, device, use_tta)
            
            for path, pred_idx, conf in zip(valid_paths, preds, confs):
                results.append({
                    "filename": path.name,
                    "category_id": CATEGORY_IDS[pred_idx],
                    "confidence": round(float(conf), 6)  # 确保转为Python float
                })
                
            processed_count += len(valid_paths)
            # print(f"进度: {processed_count}/{len(imgs)} ({(processed_count/len(imgs)*100):.1f}%)")
            
        except Exception as e:
            print(f"批量推理出错: {e}")
            # 【新增】单个图片回退处理
            for path in valid_paths:
                try:
                    # 使用原始的单张图片处理逻辑
                    img_tensor = preprocess(path, size, mean, std)
                    if device.type == "cuda":
                        img_tensor = img_tensor.half()
                    
                    # 修改：回退时也要使用 TTA，保持一致性！
                    idx, conf = predict(model, img_tensor, device, use_tta=use_tta) 
                    
                    results.append({
                        "filename": path.name,
                        "category_id": CATEGORY_IDS[idx],
                        "confidence": round(conf, 6)
                    })
                except Exception as e2:
                    # print(f"回退处理失败 {path.name}: {e2}")
                    continue
    
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 创建 DataFrame 并确保列顺序正确
    df = pd.DataFrame(results)
    
    # 确保列顺序严格一致
    df = df[['filename', 'category_id', 'confidence']]
    
    # 确保 category_id 是整数类型
    df['category_id'] = df['category_id'].astype(int)
    
    # 确保 confidence 是浮点数，保留4位小数
    df['confidence'] = df['confidence'].round(4)
    
    # 保存为 UTF-8 编码的 CSV，不带 BOM
    df.to_csv(args.output_path, index=False, encoding='utf-8')
    
    # print(f"结果已保存到 {args.output_path}，共 {len(df)} 条记录")


if __name__ == "__main__":
    main()