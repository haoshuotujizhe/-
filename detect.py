import os
from PIL import Image

def find_corrupted_images_in_train():
    """
    查找与脚本同级的 'datasets/train' 文件夹中所有损坏的图片
    （脚本需放在与 'datasets' 文件夹同级的目录下）
    """
    # 1. 自动获取 'datasets/train' 的绝对路径
    # 获取当前脚本所在目录（即与 datasets 同级的目录）
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接得到 train 图片文件夹路径
    train_img_dir = os.path.join(current_script_dir, "datasets", "train","train")
    
    # 2. 检查 train 文件夹是否存在
    if not os.path.exists(train_img_dir):
        print(f"❌ 错误：未找到 'datasets/train' 文件夹，路径：{train_img_dir}")
        print("请确认脚本与 'datasets' 文件夹在同一级目录下！")
        return []
    
    # 3. 定义支持的图片格式（覆盖常见类型）
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    corrupted_images = []  # 存储损坏图片的完整路径
    
    # 4. 遍历 train 文件夹下所有图片
    print(f"✅ 开始检查 'datasets/train' 文件夹，路径：{train_img_dir}")
    print("正在扫描图片...（耗时取决于图片数量）\n")
    
    for filename in os.listdir(train_img_dir):
        # 只处理图片格式的文件
        if filename.lower().endswith(image_extensions):
            img_full_path = os.path.join(train_img_dir, filename)
            
            try:
                # 验证图片完整性（双重校验：文件结构 + 加载能力）
                with Image.open(img_full_path) as img:
                    img.verify()  # 校验文件头和结构是否正常
                    img = Image.open(img_full_path)  # 重新打开（verify后需重开）
                    img.load()  # 加载像素数据，确认无截断
                    
            except Exception as e:
                # 捕获所有读取/验证异常，视为图片损坏
                corrupted_images.append(img_full_path)
                # 实时打印损坏图片信息（包含完整路径）
                print(f"⚠️  损坏图片：{img_full_path}")
                print(f"   错误原因：{str(e)[:100]}...\n")  # 只显示前100字符，避免输出过长
    
    # 5. 输出最终检查结果
    print("=" * 50)
    if corrupted_images:
        print(f"检查完成！共发现 {len(corrupted_images)} 张损坏图片：")
        for idx, path in enumerate(corrupted_images, 1):
            print(f"{idx}. {path}")
    else:
        print("检查完成！'datasets/train' 文件夹中所有图片均正常，无损坏文件。")
    
    return corrupted_images

if __name__ == "__main__":
    find_corrupted_images_in_train()