#!/usr/bin/env python3
"""
PCNN 模型 —— Grad-CAM 注意力热力图生成脚本

PCNN (Patch-based CNN) 使用多个 ResNet18 提取面部不同区域的特征，
通过 STN 校正和特征融合进行分类。

用法:
    python generate_gradcam_pcnn.py <图片路径>
    python generate_gradcam_pcnn.py dusk/test_0096_aligned.jpg

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
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PCNN_DIR = os.path.join(_PROJECT_ROOT, 'external', 'PCNN')

# PCNN 模型权重路径
PCNN_CHECKPOINT = os.path.join(PCNN_DIR, 'experiment', 'fer2013', 'fer2013.pth')
# PCNN 的预训练 ResNet 权重
PRETRAINED_RESNET = os.path.join(PCNN_DIR, 'models', 'resnet18_msceleb.pth')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = 224
OUTPUT_DIR = "heatmaps_pcnn"

# ImageNet 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 兼容 PCNN checkpoint 中保存的 RecorderMeter 类 ──
class RecorderMeter:
    """占位类，仅用于 checkpoint 反序列化"""
    def __init__(self, total_epoch=0):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = None
        self.epoch_accuracy = None

    def reset(self, total_epoch):
        pass

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        pass

    def plot_curve(self, save_path):
        pass


def load_pcnn_model(device):
    """加载训练好的 PCNN 模型"""
    # 切换到 PCNN 目录以便正确加载相对路径的预训练权重
    original_cwd = os.getcwd()
    os.chdir(PCNN_DIR)

    try:
        # 将 PCNN 目录加入 path
        if PCNN_DIR not in sys.path:
            sys.path.insert(0, PCNN_DIR)

        from network.models import PCNN

        # 创建模型
        model = PCNN(num_class=7, device=device)

        # 加载训练好的权重
        if os.path.exists(PCNN_CHECKPOINT):
            checkpoint = torch.load(PCNN_CHECKPOINT, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ PCNN 模型已加载 (checkpoint: {PCNN_CHECKPOINT})")
        else:
            print(f"⚠️  未找到训练权重 {PCNN_CHECKPOINT}，使用预训练权重")
    finally:
        os.chdir(original_cwd)

    model.to(device)
    model.eval()
    return model


def predict_emotion(model, input_tensor, device):
    """预测表情类别"""
    with torch.no_grad():
        x1, heads = model(input_tensor)
        probs = torch.softmax(x1, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    print(f"\n📸 预测结果: {EMOTIONS[pred.item()]} (置信度: {conf.item():.2%})")
    print("-" * 40)
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        bar = "█" * int(prob.item() * 30) + "░" * (30 - int(prob.item() * 30))
        print(f"  {emotion:10s} {bar} {prob.item():.3f}")
    print("-" * 40)
    return pred.item(), conf.item()


def reshape_transform_pcnn(tensor, height=7, width=7):
    """PCNN 的 ResNet features8 (layer4) 输出是 (B, C, H, W)，无需 reshape"""
    return tensor


def generate_gradcam_heatmap(model, input_tensor, rgb_img_float, device):
    """
    生成 PCNN 的 Grad-CAM 热力图

    注意: features8 (layer4) 在 forward 中被调用了两次（STN + 分类），
    不能用它做目标层。改用 features1 的 layer3 最后一个 conv，
    它直接从全图提取特征，只被调用一次，空间分辨率 14×14。
    """
    # ── 目标层：全图路径 features1 的 layer3，14×14 ──
    # 不能用区域路径(features2-7)，它们的特征图空间只对应局部区域
    # layer3 比 layer2 语义更强，梯度信号更集中
    target_layers = [model.features1[6][-1].conv2]

    # PCNN 的 forward 返回 (x1, heads) 元组，GradCAM 期望单个 tensor
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            x1, _ = self.base_model(x)
            return x1

    wrapped_model = ModelWrapper(model)

    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    # aug_smooth=False: 不做 TTA 平滑，保留原始激活强度
    # eigen_smooth=True: 仅去除高频棋盘格噪声
    grayscale_cam = cam(input_tensor=input_tensor,
                        eigen_smooth=True,
                        aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]

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
    parser = argparse.ArgumentParser(description="PCNN Grad-CAM 热力图生成")
    parser.add_argument('image', nargs='?', type=str, default=None,
                        help='输入图片路径')
    parser.add_argument('--model', type=str, default=PCNN_CHECKPOINT,
                        help='PCNN 模型权重路径')
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
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]
    print(f"📷 图片尺寸: {orig_w}×{orig_h}")

    # ---- 加载模型 ----
    print(f"🖥️  设备: {DEVICE}")
    model = load_pcnn_model(DEVICE)

    # ---- 预处理（与 PCNN 训练时一致） ----
    rgb_img_float = rgb_img / 255.0
    rgb_img_resized = cv2.resize(rgb_img_float, (INPUT_SIZE, INPUT_SIZE))

    # ImageNet 归一化
    input_tensor = (rgb_img_resized - MEAN) / STD
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

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    emotion_name = EMOTIONS[predicted_idx].lower()

    # 热力图叠加结果
    output_filename = f"pcnn_gradcam_{emotion_name}_{base_name}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ 热力图已保存至: {output_path}")


if __name__ == "__main__":
    main()
