# generate_gradcam_cnn.py
"""
CNN 面部表情识别 —— Grad-CAM 注意力热力图生成脚本

用法:
    python generate_gradcam_cnn.py <图片路径>
    python generate_gradcam_cnn.py images/test_face.jpg

依赖:
    pip install pytorch-grad-cam
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from scipy.ndimage import gaussian_filter

# ---------- 导入 CNN 模型 ----------
# 确保可以从 utilities 目录导入
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utilities.train_cnn import FER_CNN

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = "Models/CNN/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = 224
OUTPUT_DIR = "heatmaps"


def load_cnn_model(model_path, device):
    """加载训练好的 CNN 模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_args = checkpoint.get('model_args', {'num_classes': 7, 'dropout_rate': 0.4})
    model = FER_CNN(**model_args)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    print(f"✅ 模型已加载 (准确率: {checkpoint.get('accuracy', 'N/A')})")
    return model


def predict_emotion(model, input_tensor, device):
    """预测表情类别"""
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    print(f"\n📸 预测结果: {EMOTIONS[pred.item()]} (置信度: {conf.item():.2%})")
    print("-" * 40)
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        bar = "█" * int(prob.item() * 30) + "░" * (30 - int(prob.item() * 30))
        print(f"  {emotion:10s} {bar} {prob.item():.3f}")
    print("-" * 40)
    return pred.item(), conf.item()


def generate_gradcam_heatmap(model, input_tensor, rgb_img_float, device):
    """
    生成 CNN 的 Grad-CAM 热力图
    CNN 不需要 reshape_transform，直接使用最后一个卷积层即可
    """
    # ── 目标层：block4_conv2（最后一个卷积层，语义信息最丰富） ──
    target_layers = [model.block4_conv2]

    cam = GradCAM(model=model, target_layers=target_layers)

    # eigen_smooth 消除棋盘格噪声，aug_smooth 增强鲁棒性
    grayscale_cam = cam(input_tensor=input_tensor,
                        eigen_smooth=True,
                        aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]  # 取出单张图 (H, W)

    # 获取原图尺寸
    orig_h, orig_w = rgb_img_float.shape[:2]

    # 将低分辨率热力图 (14×14) 放大到原图尺寸
    heatmap_resized = cv2.resize(grayscale_cam, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)

    # 高斯模糊平滑，sigma 根据图片大小自适应
    sigma = max(orig_w, orig_h) / 120.0
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=sigma)

    # 叠加原图与热力图
    visualization = show_cam_on_image(rgb_img_float, heatmap_resized, use_rgb=True)

    return visualization, grayscale_cam, heatmap_resized


def main():
    parser = argparse.ArgumentParser(
        description="CNN 面部表情识别 Grad-CAM 热力图生成"
    )
    parser.add_argument('image', nargs='?', type=str, default=None,
                        help='输入图片路径')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='CNN 模型权重路径')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='热力图输出目录')
    parser.add_argument('--no-smooth', action='store_true',
                        help='禁用高斯平滑')
    args = parser.parse_args()

    # ---- 获取图片路径 ----
    if args.image:
        image_path = args.image
    else:
        image_path = input("请输入图片路径: ").strip().strip("'\"")

    # ---- 读取图片（支持中文路径） ----
    rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb_img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]
    print(f"📷 图片尺寸: {orig_w}×{orig_h}")

    # ---- 检查模型 ----
    if not os.path.exists(args.model):
        print(f"\n⚠️  模型文件不存在: {args.model}")
        print("请先训练 CNN 模型:")
        print("  python utilities/train_cnn.py --epochs 50")
        print("\n或使用 Swin Transformer 模型:")
        print("  python generate_gradcam.py <图片路径>")
        sys.exit(1)

    # ---- 加载模型 ----
    print(f"🖥️  设备: {DEVICE}")
    model = load_cnn_model(args.model, DEVICE)

    # ---- 预处理（与训练时一致） ----
    rgb_img_float = rgb_img / 255.0
    rgb_img_resized = cv2.resize(rgb_img_float, (INPUT_SIZE, INPUT_SIZE))

    # 构建输入张量（归一化使用 ImageNet 统计量）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = (rgb_img_resized - mean) / std
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    # ---- 预测表情 ----
    predicted_idx, confidence = predict_emotion(model, input_tensor, DEVICE)

    # ---- 生成 Grad-CAM 热力图 ----
    print("🔥 正在生成 Grad-CAM 热力图...")
    visualization, raw_cam, smooth_cam = generate_gradcam_heatmap(
        model, input_tensor, rgb_img_float, DEVICE
    )

    # ---- 保存结果 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # 热力图叠加结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    emotion_name = EMOTIONS[predicted_idx].lower()
    output_filename = f"cnn_gradcam_{emotion_name}_{base_name}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ 热力图已保存至: {output_path}")

    # 纯热力图（无原图叠加）
    heatmap_only_filename = f"cnn_heatmap_{emotion_name}_{base_name}.png"
    heatmap_only_path = os.path.join(args.output_dir, heatmap_only_filename)
    heatmap_colored = cv2.applyColorMap(
        (smooth_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(heatmap_only_path, heatmap_colored)
    print(f"✅ 纯热力图已保存至: {heatmap_only_path}")


if __name__ == "__main__":
    main()
