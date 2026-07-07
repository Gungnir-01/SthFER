#!/usr/bin/env python
"""
Swin Transformer 四阶段注意力热力图生成脚本
============================================
对 kita 文件夹中的图片，使用 Swin Transformer 模型的四个 Stage
分别生成 Grad-CAM 注意力热力图，展示模型从浅层到深层的关注区域演变。

四个阶段对应 Swin 的 4 个 Stage：
  Stage 0 (浅层): 56×56 特征图 — 纹理/边缘
  Stage 1:        28×28 特征图 — 局部形状
  Stage 2:        14×14 特征图 — 部件/模式
  Stage 3 (深层):  7×7 特征图 — 全局语义

输出（保存在 yikuyou/ 目录）：
  - {图片名}_stage0_heatmap.png ... stage3_heatmap.png  (热力图可视化)
  - {图片名}_stage0_matrix.npy  ... stage3_matrix.npy   (原始热力矩阵)
  - {图片名}_comparison.png                             (四阶段对比图)
"""

import torch
import timm
import cv2
import numpy as np
import os
import sys
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
MODEL_PATH = "Models/Swin_Transformer/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_DIR = "kita"
OUTPUT_DIR = "yikuyou"

# 四个 Stage 的目标层及对应的空间尺寸
# model.layers 是 nn.Sequential，包含 4 个 SwinTransformerStage
# 每个 Stage 取最后一个 block 的 norm2 层
# 路径格式: layers.<stage_idx>.blocks.<block_idx>.<attr>
STAGE_CONFIG = {
    "stage0": {
        "label": "Stage 0 (Texture/Edges)",
        "target_layer": "layers.0.blocks.-1.norm2",
        "spatial_size": 56,
        "color": "Blues",
    },
    "stage1": {
        "label": "Stage 1 (Local Shapes)",
        "target_layer": "layers.1.blocks.-1.norm2",
        "spatial_size": 28,
        "color": "Greens",
    },
    "stage2": {
        "label": "Stage 2 (Parts/Patterns)",
        "target_layer": "layers.2.blocks.-1.norm2",
        "spatial_size": 14,
        "color": "Oranges",
    },
    "stage3": {
        "label": "Stage 3 (Global Semantics)",
        "target_layer": "layers.3.blocks.-1.norm2",
        "spatial_size": 7,
        "color": "Reds",
    },
}


def create_reshape_transform(target_spatial_size):
    """
    创建针对特定空间尺寸的 Swin reshape_transform。
    Swin 中间层输出为 (B, L, C) 格式，需转为 (B, C, H, W)。
    """
    def reshape_transform(tensor, height=target_spatial_size, width=target_spatial_size):
        if tensor.dim() == 4:
            return tensor
        elif tensor.dim() == 3:
            B, L, C = tensor.shape
            HW = int(math.sqrt(L))
            if HW * HW != L:
                # 如果 L 不是完全平方数，尝试用给定的 height/width
                if height * width == L:
                    HW_h, HW_w = height, width
                else:
                    # 回退：尝试多种可能的分解
                    result = None
                    for h in range(1, int(math.sqrt(L)) + 10):
                        if L % h == 0:
                            w = L // h
                            result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
                            break
                    if result is not None:
                        return result
                    raise ValueError(f"无法将序列长度 {L} 重塑为空间格式")
            else:
                HW_h, HW_w = HW, HW
            result = tensor.reshape(B, HW_h, HW_w, C).permute(0, 3, 1, 2)
            return result
        else:
            return tensor
    return reshape_transform


def resolve_target_layer(model, layer_spec):
    """
    解析目标层路径，如 'layers.0.blocks.-1.norm2'
    支持：
      - 数字索引：访问 Sequential/ModuleList 的第 N 个元素
      - '-1' 索引：访问最后一个元素
      - 字符串：通过 getattr 访问
    """
    parts = layer_spec.split('.')
    obj = model
    for i, part in enumerate(parts):
        if part == '-1':
            # 获取当前 Sequential/ModuleList 的最后一个元素
            if hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
                obj = obj[len(obj) - 1]
            elif hasattr(obj, '__getitem__'):
                obj = obj[-1]
            else:
                raise ValueError(f"无法对 {type(obj).__name__} 使用 -1 索引 (路径: {layer_spec}, 当前部分: {part})")
        elif part.lstrip('-').isdigit():
            # 纯数字索引
            obj = obj[int(part)]
        else:
            # 先尝试 getattr，失败则尝试数字索引
            try:
                obj = getattr(obj, part)
            except AttributeError:
                try:
                    obj = obj[int(part)]
                except (ValueError, TypeError, IndexError):
                    raise AttributeError(f"无法解析路径 '{layer_spec}' 中的 '{part}': "
                                        f"{type(obj).__name__} 没有该属性或索引")
    return obj


def load_swin_model(model_path, device):
    """加载 Swin Transformer 模型"""
    model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ 模型已加载: {model_path}")
    return model


def predict_emotion(model, input_tensor):
    """预测表情并返回类别名和置信度"""
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return EMOTIONS[pred.item()], conf.item(), probs.cpu().numpy()


def generate_stage_heatmap(model, input_tensor, target_layer, reshape_transform_fn, rgb_img_float):
    """
    生成单个 Stage 的 Grad-CAM 热力图。

    返回:
        visualization: RGB 叠加图 (uint8)
        grayscale_cam: 原始低分辨率热力图 (H, W) numpy
        heatmap_resized: 放大到原图尺寸的热力图 (H, W) numpy
    """
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform_fn,
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


def create_comparison_figure(rgb_img_float, heatmaps_info, predicted_label, confidence):
    """
    Create a 4-stage comparison figure (2×3 layout: original + 4 heatmaps + probability bars)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Predicted: {predicted_label} ({confidence:.1%})  |  Swin 4-Stage Attention Heatmaps",
                 fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(rgb_img_float)
    axes[0, 0].set_title("Original Image", fontsize=13)
    axes[0, 0].axis('off')

    # Four stage heatmaps
    stage_keys = ["stage0", "stage1", "stage2", "stage3"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

    for (row, col), key in zip(positions, stage_keys):
        info = heatmaps_info[key]
        axes[row, col].imshow(info['visualization'])
        axes[row, col].set_title(info['label'], fontsize=12)
        axes[row, col].axis('off')

    # Class probability distribution
    probs = heatmaps_info['probs']
    class_names = heatmaps_info['class_names']
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#C9B1FF', '#FF8C42', '#A0A0A0']
    bars = axes[1, 2].barh(class_names, probs * 100, color=colors)
    axes[1, 2].set_xlim(0, 100)
    axes[1, 2].set_xlabel("Probability (%)", fontsize=11)
    axes[1, 2].set_title("Class Probabilities", fontsize=13)
    for bar, p in zip(bars, probs):
        axes[1, 2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{p*100:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 加载模型 ──
    print(f"🖥️  设备: {DEVICE}")
    model = load_swin_model(MODEL_PATH, DEVICE)

    # ── 准备各 Stage 的目标层和 reshape_transform ──
    stage_targets = {}
    for stage_key, cfg in STAGE_CONFIG.items():
        target_layer = resolve_target_layer(model, cfg["target_layer"])
        reshape_fn = create_reshape_transform(cfg["spatial_size"])
        stage_targets[stage_key] = {
            "target_layer": target_layer,
            "reshape_fn": reshape_fn,
            "label": cfg["label"],
        }
        print(f"   🎯 {stage_key}: {cfg['target_layer']} → {type(target_layer).__name__}")

    # ── 遍历 kita 文件夹中的图片 ──
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ])

    if not image_files:
        print(f"❌ 在 {INPUT_DIR}/ 中未找到图片文件!")
        sys.exit(1)

    print(f"\n📷 找到 {len(image_files)} 张图片待处理\n")

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        img_name = os.path.splitext(img_file)[0]
        print(f"{'='*60}")
        print(f"🔍 处理: {img_file}")

        # 读取图片（支持中文路径）
        rgb_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"   ⚠️ 无法读取: {img_path}，跳过")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_img.shape[:2]
        print(f"   原始尺寸: {orig_w}×{orig_h}")

        # 预处理
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = rgb_img / 255.0
        input_tensor = torch.from_numpy(rgb_img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        # 预测
        predicted_label, confidence, probs = predict_emotion(model, input_tensor)
        print(f"   预测表情: {predicted_label} (置信度: {confidence:.2%})")

        # ── 为每个 Stage 生成热力图 ──
        heatmaps_info = {
            'probs': probs,
            'class_names': EMOTIONS,
        }

        for stage_key in ["stage0", "stage1", "stage2", "stage3"]:
            cfg = stage_targets[stage_key]
            print(f"   ⏳ 生成 {stage_key} 热力图...")

            visualization, grayscale_cam, heatmap_resized = generate_stage_heatmap(
                model=model,
                input_tensor=input_tensor,
                target_layer=cfg["target_layer"],
                reshape_transform_fn=cfg["reshape_fn"],
                rgb_img_float=rgb_img_float,
            )

            heatmaps_info[stage_key] = {
                'visualization': visualization,
                'grayscale_cam': grayscale_cam,
                'heatmap_resized': heatmap_resized,
                'label': cfg["label"],
            }

            # 保存热力图可视化
            heatmap_img_path = os.path.join(OUTPUT_DIR, f"{img_name}_{stage_key}_heatmap.png")
            cv2.imwrite(heatmap_img_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            # 保存原始热力矩阵（低分辨率版本，保留模型关注的精确空间分布）
            matrix_path = os.path.join(OUTPUT_DIR, f"{img_name}_{stage_key}_matrix.npy")
            np.save(matrix_path, grayscale_cam)

            # 同时保存放大到原图尺寸的热力矩阵
            matrix_full_path = os.path.join(OUTPUT_DIR, f"{img_name}_{stage_key}_matrix_full.npy")
            np.save(matrix_full_path, heatmap_resized)

            print(f"      ✅ 已保存: {stage_key}_heatmap.png + {stage_key}_matrix.npy + matrix_full.npy")

        # ── 生成四阶段对比图 ──
        print(f"   📊 生成四阶段对比图...")
        fig = create_comparison_figure(rgb_img_float, heatmaps_info, predicted_label, confidence)
        comparison_path = os.path.join(OUTPUT_DIR, f"{img_name}_comparison.png")
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"      ✅ 对比图已保存: {img_name}_comparison.png")

    print(f"\n{'='*60}")
    print(f"🏁 全部完成! 结果保存在: {OUTPUT_DIR}/")
    print(f"   文件列表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"     📄 {f} ({fsize/1024:.1f} KB)")


if __name__ == "__main__":
    main()
