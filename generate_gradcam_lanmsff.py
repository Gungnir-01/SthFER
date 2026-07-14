#!/usr/bin/env python3
"""
LANMSFF 模型 —— Grad-CAM 注意力热力图生成脚本

用法:
    python generate_gradcam_lanmsff.py <图片路径>
    python generate_gradcam_lanmsff.py dusk/test_0096_aligned.jpg

依赖:
    pip install pytorch-grad-cam
"""

import torch
import cv2
import numpy as np
import os
import sys
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# ---- 导入 LANMSFF 模型 ----
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from external.LANMSFF.model import LANMSFF

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
MODEL_PATH = "Models/LANMSFF/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = 64  # LANMSFF 使用 64×64 输入
OUTPUT_DIR = "heatmaps_lanmsff"


def load_model(model_path, device):
    """加载训练好的 LANMSFF 模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_args = checkpoint.get('model_args', {'num_classes': 7, 'input_channels': 1})
    model = LANMSFF(**model_args)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    print(f"✅ LANMSFF 模型已加载 (epoch: {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
    print(f"   参数: num_classes={model_args['num_classes']}, input_channels={model_args['input_channels']}")
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
    生成 LANMSFF 的 Grad-CAM 热力图
    目标层: block3 的最后一个 conv + block4 的最后一个 conv
    """
    # ── 目标层（优化后：浅层 + 中层，更高空间分辨率） ──
    # block2_conv: MassAtt 注意力融合后的 1×1 卷积，空间分辨率 32×32，保留纹理
    # block3[7]: block3 最后一个 3×3 卷积（maxpool 前），空间分辨率 16×16，语义更强
    target_layers = [
        model.block2_conv,       # 32×32，浅层纹理 + 注意力信息
        model.block3[7],         # 16×16，中层语义
    ]

    cam = GradCAM(model=model, target_layers=target_layers)

    # eigen_smooth 消除棋盘格噪声，aug_smooth 增强鲁棒性
    grayscale_cam = cam(input_tensor=input_tensor,
                        eigen_smooth=True,
                        aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]  # 取出单张图 (H, W)

    # 获取原图尺寸
    orig_h, orig_w = rgb_img_float.shape[:2]

    # 将低分辨率热力图放大到原图尺寸
    heatmap_resized = cv2.resize(grayscale_cam, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)

    # 高斯模糊平滑
    sigma = max(orig_w, orig_h) / 120.0
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=sigma)

    # 叠加原图与热力图（使用原始 Grad-CAM 值，保留真实注意力分布）
    visualization = show_cam_on_image(rgb_img_float, heatmap_resized, use_rgb=True)

    return visualization, grayscale_cam, heatmap_resized


def main():
    parser = argparse.ArgumentParser(description="LANMSFF Grad-CAM 热力图生成")
    parser.add_argument('image', nargs='?', type=str, default=None,
                        help='输入图片路径')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='模型权重路径')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    args = parser.parse_args()

    # ---- 获取图片路径 ----
    if args.image:
        image_path = args.image
    else:
        image_path = input("请输入图片路径: ").strip().strip("'\"")

    # ---- 读取图片（支持中文路径） ----
    rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb_img is None:
        # 尝试灰度读取
        rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if rgb_img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        # 灰度转 RGB 用于显示
        rgb_img_display = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        is_gray = True
    else:
        rgb_img_display = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        is_gray = False

    orig_h, orig_w = rgb_img_display.shape[:2]
    print(f"📷 图片尺寸: {orig_w}×{orig_h}, 灰度: {is_gray}")

    # ---- 检查模型 ----
    if not os.path.exists(args.model):
        print(f"⚠️  模型文件不存在: {args.model}")
        sys.exit(1)

    # ---- 加载模型 ----
    print(f"🖥️  设备: {DEVICE}")
    model = load_model(args.model, DEVICE)

    # ---- 预处理 ----
    # LANMSFF 使用灰度图, 64×64
    if is_gray:
        gray_img = rgb_img
    else:
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    gray_resized = cv2.resize(gray_img, (INPUT_SIZE, INPUT_SIZE))
    # 归一化到 [0, 1]
    input_tensor = torch.from_numpy(gray_resized / 255.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    # shape: (1, 1, 64, 64)

    # ---- 预测表情 ----
    predicted_idx, confidence = predict_emotion(model, input_tensor, DEVICE)

    # ---- 生成 Grad-CAM 热力图 ----
    print("🔥 正在生成 Grad-CAM 热力图...")
    rgb_float = rgb_img_display / 255.0
    visualization, raw_cam, smooth_cam = generate_gradcam_heatmap(
        model, input_tensor, rgb_float, DEVICE
    )

    # ---- 保存结果 ----
    os.makedirs(args.output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    emotion_name = EMOTIONS[predicted_idx].lower()

    # 热力图叠加结果
    output_filename = f"lanmsff_gradcam_{emotion_name}_{base_name}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ 热力图已保存至: {output_path}")


if __name__ == "__main__":
    main()
