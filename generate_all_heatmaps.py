#!/usr/bin/env python
"""
三模型注意力热力图批量生成脚本
=================================
对 yukino/ 中每张图片，分别用 CNN、ViT、Swin Transformer 三个模型
生成 Grad-CAM 注意力热力图，结果统一保存在 yushita/ 中。

输出文件命名规则:
  {图片名}_{模型名}_heatmap.png      — 热力图叠加原图
  {图片名}_{模型名}_heatmap_only.png — 纯热力图（无原图）
  {图片名}_{模型名}_matrix.npy       — 热力矩阵
  {图片名}_{模型名}_matrix_full.npy  — 全尺寸热力矩阵

模型及目标层:
  CNN:  FER_CNN → block4_conv2
  ViT:  CustomViTWithHead → vit.transformer.layers[-1].to_qkv
  Swin: swin_small → layers.3.blocks.-1.norm2
"""

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import os
import sys
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

INPUT_DIR = "yukino"
OUTPUT_DIR = "yushita"

MODEL_PATHS = {
    "cnn":  "Models/CNN/best_model.pth",
    "vit":  "Models/ViT_Model/best_model_vit.pth",
    "swin": "Models/Swin_Transformer/best_model.pth",
}

# ── 导入自定义模型 ──
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utilities.train_cnn import FER_CNN
from utilities.vit_model import CustomViTWithHead


# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════

def load_cnn_model(model_path, device):
    """加载 CNN 模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get('model_args', {'num_classes': 7, 'dropout_rate': 0.4})
    model = FER_CNN(**model_args)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def load_vit_model(model_path, device):
    """加载 ViT 模型，并包装为 Grad-CAM 兼容格式"""
    model = CustomViTWithHead(
        image_size=224, patch_size=32, num_classes=7,
        dim=1024, depth=6, heads=16, mlp_dim=2048
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Grad-CAM 包装器：取 CLS token 输出
    class ViTGradCAMWrapper(nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit_model = vit_model
        def forward(self, x):
            out = self.vit_model(x)       # [B, num_patches+1, dim]
            return out[:, 0, :]           # [B, dim]

    wrapped = ViTGradCAMWrapper(model)
    wrapped.to(device)
    wrapped.eval()
    return model, wrapped


def load_swin_model(model_path, device):
    """加载 Swin Transformer 模型"""
    model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
# Reshape Transforms
# ═══════════════════════════════════════════════════════════

def vit_reshape_transform(tensor, height=14, width=14):
    """ViT: (B, L, C) → (B, C, H, W)，去掉 CLS token"""
    if tensor.dim() == 4:
        return tensor
    elif tensor.dim() == 3:
        B, L, C = tensor.shape
        L = L - 1  # 去掉 CLS
        hw = int(math.sqrt(L))
        if hw * hw != L:
            raise ValueError(f"序列长度 {L} 不是完全平方数")
        return tensor[:, 1:, :].reshape(B, hw, hw, C).permute(0, 3, 1, 2)
    return tensor


def swin_reshape_transform(tensor, height=7, width=7):
    """Swin: (B, L, C) → (B, C, H, W)"""
    if tensor.dim() == 4:
        return tensor
    elif tensor.dim() == 3:
        B, L, C = tensor.shape
        HW = int(math.sqrt(L))
        if HW * HW != L:
            raise ValueError(f"序列长度 {L} 不是完全平方数")
        return tensor.reshape(B, HW, HW, C).permute(0, 3, 1, 2)
    return tensor


# ═══════════════════════════════════════════════════════════
# 热力图生成核心
# ═══════════════════════════════════════════════════════════

def generate_heatmap(model, input_tensor, target_layers, rgb_img_float,
                     reshape_transform=None):
    """
    通用 Grad-CAM 热力图生成

    返回:
        visualization:   叠加原图的 RGB 图 (uint8)
        grayscale_cam:   低分辨率热力图 (H, W) numpy
        heatmap_resized: 全尺寸热力图 (orig_H, orig_W) numpy
    """
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    orig_h, orig_w = rgb_img_float.shape[:2]

    # 放大到原图尺寸
    heatmap_resized = cv2.resize(grayscale_cam, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)
    # 高斯模糊平滑
    sigma = max(orig_w, orig_h) / 80.0
    heatmap_resized = gaussian_filter(heatmap_resized, sigma=sigma)

    # 叠加
    visualization = show_cam_on_image(rgb_img_float, heatmap_resized, use_rgb=True)

    return visualization, grayscale_cam, heatmap_resized


# ═══════════════════════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def predict_cnn(model, input_tensor):
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)
    return EMOTIONS[pred.item()], conf.item()


@torch.no_grad()
def predict_vit(wrapped_model, input_tensor):
    outputs = wrapped_model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)
    return EMOTIONS[pred.item()], conf.item()


@torch.no_grad()
def predict_swin(model, input_tensor):
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)
    return EMOTIONS[pred.item()], conf.item()


# ═══════════════════════════════════════════════════════════
# 图像预处理
# ═══════════════════════════════════════════════════════════

def preprocess_image(rgb_img, device):
    """
    预处理图片为模型输入张量
    rgb_img: uint8 RGB 图像
    返回: (input_tensor, rgb_img_float)
    """
    rgb_img_float = rgb_img / 255.0
    rgb_img_resized = cv2.resize(rgb_img_float, (224, 224))

    # ImageNet 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = (rgb_img_resized - mean) / std
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)

    return input_tensor, rgb_img_float


# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════

def save_results(img_name, model_name, visualization, grayscale_cam, heatmap_resized):
    """保存热力图 PNG + 矩阵 NPY"""
    # 叠加原图的热力图
    overlay_path = os.path.join(OUTPUT_DIR, f"{img_name}_{model_name}_heatmap.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # 纯热力图（无原图叠加）
    heatmap_only = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_only_path = os.path.join(OUTPUT_DIR, f"{img_name}_{model_name}_heatmap_only.png")
    cv2.imwrite(heatmap_only_path, heatmap_only)

    # 热力矩阵
    matrix_path = os.path.join(OUTPUT_DIR, f"{img_name}_{model_name}_matrix.npy")
    np.save(matrix_path, grayscale_cam)

    matrix_full_path = os.path.join(OUTPUT_DIR, f"{img_name}_{model_name}_matrix_full.npy")
    np.save(matrix_full_path, heatmap_resized)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🖥️  设备: {DEVICE}\n")

    # ── 加载全部模型 ──
    print("=" * 60)
    print("📥 加载模型...")

    # CNN
    print("\n[1/3] CNN 模型")
    cnn_model = load_cnn_model(MODEL_PATHS["cnn"], DEVICE)
    cnn_target = [cnn_model.block4_conv2]
    print(f"   ✅ CNN 已加载 | 目标层: block4_conv2")

    # ViT
    print("\n[2/3] ViT 模型")
    vit_raw, vit_wrapped = load_vit_model(MODEL_PATHS["vit"], DEVICE)
    vit_target = [vit_raw.vit.transformer.layers[-1][0].to_qkv]
    print(f"   ✅ ViT 已加载 | 目标层: transformer.layers[-1].to_qkv")

    # Swin
    print("\n[3/3] Swin 模型")
    swin_model = load_swin_model(MODEL_PATHS["swin"], DEVICE)
    swin_target = [swin_model.layers[3].blocks[-1].norm2]
    print(f"   ✅ Swin 已加载 | 目标层: layers[3].blocks[-1].norm2")

    # ── 扫描图片 ──
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ])
    if not image_files:
        print(f"\n❌ 在 {INPUT_DIR}/ 中未找到图片!")
        sys.exit(1)
    print(f"\n📷 找到 {len(image_files)} 张图片待处理\n")

    # ── 逐张处理 ──
    total = len(image_files)
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(INPUT_DIR, img_file)
        img_name = os.path.splitext(img_file)[0]
        print(f"[{idx}/{total}] 🔍 {img_file}")

        # 读取图片
        rgb_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"   ⚠️ 无法读取，跳过")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_img.shape[:2]
        print(f"   尺寸: {orig_w}×{orig_h}")

        # 预处理
        input_tensor, rgb_img_float = preprocess_image(rgb_img, DEVICE)

        # ── CNN ──
        label_cnn, conf_cnn = predict_cnn(cnn_model, input_tensor)
        print(f"   [CNN]  预测: {label_cnn} ({conf_cnn:.2%})")
        vis, raw, full = generate_heatmap(
            cnn_model, input_tensor, cnn_target, rgb_img_float
        )
        save_results(img_name, "cnn", vis, raw, full)

        # ── ViT ──
        label_vit, conf_vit = predict_vit(vit_wrapped, input_tensor)
        print(f"   [ViT]  预测: {label_vit} ({conf_vit:.2%})")
        vis, raw, full = generate_heatmap(
            vit_wrapped, input_tensor, vit_target, rgb_img_float,
            reshape_transform=vit_reshape_transform
        )
        save_results(img_name, "vit", vis, raw, full)

        # ── Swin ──
        label_swin, conf_swin = predict_swin(swin_model, input_tensor)
        print(f"   [Swin] 预测: {label_swin} ({conf_swin:.2%})")
        vis, raw, full = generate_heatmap(
            swin_model, input_tensor, swin_target, rgb_img_float,
            reshape_transform=swin_reshape_transform
        )
        save_results(img_name, "swin", vis, raw, full)

        print(f"   ✅ 已保存 3 组热力图\n")

    # ── 汇总 ──
    print("=" * 60)
    count_png = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    count_npy = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    print(f"🏁 全部完成!")
    print(f"   图片数量: {total}")
    print(f"   每张 × 4 输出 (heatmap.png + heatmap_only.png + matrix.npy × 2) = {total * 3 * 4} 文件")
    print(f"   实际 PNG: {count_png} | NPY: {count_npy}")
    print(f"   保存路径: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
