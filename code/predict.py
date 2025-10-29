import os
import argparse
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from pathlib import Path


CATEGORY_IDS = [
    # 164-245 范围
    164, 165, 166, 167, 169, 171, 172, 173, 174, 176, 177, 178, 179, 180,
    183, 184, 185, 186, 188, 189, 190, 192, 193, 194, 195, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    243, 244, 245,
    # 1734-1833 范围
    1734, 1743, 1747, 1749, 1750, 1751, 1759, 1765, 1770, 1772, 1774, 1776,
    1777, 1780, 1784, 1785, 1786, 1789, 1796, 1797, 1801, 1805, 1806, 1808,
    1818, 1827, 1833
]

def load_config(config_path="config.json"):
    """读取配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 预测代码（predict.py）中修改 load_model 函数：
def load_model(model_path, num_classes, device):
    """加载训练好的模型(使用与训练时一致的Mymodel结构)"""
    # 从 model 模块导入 build_model（与训练时一致的模型构建函数）
    from model import build_model
    # 构建与训练时相同的模型结构（Mymodel类）
    model = build_model({"num_classes": num_classes, "use_pretrained": False})  # 不加载预训练权重，避免冲突
    # 加载保存的参数
    state_dict = torch.load(model_path, map_location=device, weights_only=True)  # 加上weights_only=True更安全
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# def load_model(model_path, num_classes, device):
#     """加载训练好的模型"""
#     model = models.efficientnet_b7(weights=None)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = torch.nn.Linear(in_features, num_classes)

#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model

def preprocess_image(image_path, input_size=(456, 456), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """对单张图片进行预处理（与训练保持一致）"""
    transform = transforms.Compose([
        transforms.Resize((int(input_size[0] * 1.05), int(input_size[1] * 1.05))),  # ✅ 与验证集一致
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device, use_tta=True):
    """单张图片预测（6 种 TTA 是最优配置）"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if use_tta:
            # ✅ 最优 TTA：6 种变换（平衡效果和速度）
            outputs_list = []
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            
            # 1. 原图
            outputs_list.append(model(image_tensor))
            
            # 2. 水平翻转
            image_h_flip = torch.flip(image_tensor, dims=[3])
            outputs_list.append(model(image_h_flip))
            
            # 3. 垂直翻转
            image_v_flip = torch.flip(image_tensor, dims=[2])
            outputs_list.append(model(image_v_flip))
            
            # 4. 水平+垂直翻转
            image_hv_flip = torch.flip(image_tensor, dims=[2, 3])
            outputs_list.append(model(image_hv_flip))
            
            # 5. 缩小 0.9 倍（保留最有效的尺度变换）
            image_scale_down = F.interpolate(image_tensor, size=(int(h*0.9), int(w*0.9)), 
                                            mode='bilinear', align_corners=False)
            image_scale_down = F.interpolate(image_scale_down, size=(h, w), 
                                            mode='bilinear', align_corners=False)
            outputs_list.append(model(image_scale_down))
            
            # 6. 放大 1.1 倍（保留最有效的尺度变换）
            image_scale_up = F.interpolate(image_tensor, size=(int(h*1.1), int(w*1.1)), 
                                          mode='bilinear', align_corners=False)
            crop_h = (image_scale_up.shape[2] - h) // 2
            crop_w = (image_scale_up.shape[3] - w) // 2
            image_scale_up = image_scale_up[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
            outputs_list.append(model(image_scale_up))
            
            # ✅ 平均所有预测
            outputs = torch.mean(torch.stack(outputs_list), dim=0)
        else:
            outputs = model(image_tensor)
        
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    return int(pred.item()), float(conf.item())

def main():
    parser = argparse.ArgumentParser(description="花卉分类模型预测")
    parser.add_argument("test_img_dir", type=str, help="测试图片目录")
    parser.add_argument("output_path", type=str, help="预测结果 CSV 文件路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_path = project_root / "model" / "config.json"
    config = load_config(config_path)

    model_path = project_root / "model" / "best_model.pth"
    model = load_model(model_path, config["num_classes"], device)

    image_dir = Path(args.test_img_dir)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    # ✅ 从 config 读取预处理参数
    input_size = tuple(config["input_size"])
    mean = config["mean"]
    std = config["std"]
    use_tta = config.get("use_tta", True)  # 从配置读取是否启用 TTA

    results = []
    for img_path in image_paths:
        img_tensor = preprocess_image(img_path, input_size=input_size, mean=mean, std=std)
        pred_index, confidence = predict(model, img_tensor, device, use_tta=use_tta)
        
        category_id = CATEGORY_IDS[pred_index]
        
        results.append({
            "filename": img_path.name,
            "category_id": category_id,
            "confidence": confidence
        })

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False, encoding="utf-8")

    print(f"✅ 预测完成，共处理 {len(results)} 张图片。结果已保存至：{args.output_path}")
    print(f"   TTA: {'启用' if use_tta else '禁用'}")

if __name__ == "__main__":
    main()
