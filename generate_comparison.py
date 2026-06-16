"""
TDR 对比热力图生成脚本
输入一张人脸图片，输出三张对比图：
  1. 无 TDR 模块的注意力热力图（Swin baseline）
  2. 有 TDR 模块的注意力热力图（Swin + TDR）
  3. 细节差异图（突出 TDR 恢复的面部纹理细节）

用法:
    python generate_comparison.py <图片路径>
    python generate_comparison.py images/03204.jpg
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# 导入 TDR 模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilities.tdr import SwinTDR

# ══════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════
SWIN_MODEL_PATH = "Models/Swin_Transformer/best_model.pth"
TDR_MODEL_PATH = "Models/Swin_Transformer/best_model_tdr.pth"  # TDR 训练后保存
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
OUTPUT_DIR = "comparison"


def swin_reshape_transform(tensor, height=7, width=7):
    """Swin Transformer 的 Grad-CAM reshape"""
    if tensor.dim() == 4:
        return tensor
    elif tensor.dim() == 3:
        B, L, C = tensor.shape
        HW = int(math.sqrt(L))
        if HW * HW != L:
            raise ValueError(f"序列长度 {L} 不是完全平方数")
        return tensor.reshape(B, HW, HW, C).permute(0, 3, 1, 2)
    return tensor


def load_swin_model(model_path, device):
    """加载标准 Swin Transformer（无 TDR）"""
    import timm
    model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_tdr_model(swin_path, tdr_path, device):
    """
    加载 Swin+TDR 模型
    优先使用 TDR 专用权重，否则从 Swin checkpoint 迁移 backbone
    """
    model = SwinTDR(
        model_name='swin_small_patch4_window7_224',
        num_classes=7,
        pretrained_backbone=False,
        pretrained_swin_path=swin_path,   # 迁移 backbone
        tdr_hidden_dim=256,
        dropout_rate=0.5,
    )

    if os.path.exists(tdr_path):
        print(f"✅ 加载 TDR 专用权重: {tdr_path}")
        state_dict = torch.load(tdr_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️  TDR 权重不存在 ({tdr_path})，TDR 模块随机初始化，需训练")

    model.to(device)
    model.eval()
    return model


def generate_gradcam_swin(model, input_tensor, rgb_img_float, device):
    """标准 Swin 的 Grad-CAM"""
    target_layers = [
        model.layers[-1].blocks[-1].mlp.fc2,
        model.norm,
    ]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=swin_reshape_transform,
    )
    grayscale_cam = cam(
        input_tensor=input_tensor,
        eigen_smooth=True,
        aug_smooth=True,
    )
    grayscale_cam = grayscale_cam[0, :]

    orig_h, orig_w = rgb_img_float.shape[:2]
    heatmap = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    sigma = max(orig_w, orig_h) / 80.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    visualization = show_cam_on_image(rgb_img_float, heatmap, use_rgb=True)
    return visualization, heatmap


def generate_gradcam_tdr(model, input_tensor, rgb_img_float, device):
    """Swin+TDR 的 Grad-CAM"""
    # 目标层: TDR 的空间门控 + 融合层最后一个卷积
    # 这些层能反映 TDR 模块关注的空间位置
    target_layers = [
        model.tdr.fusion[-2],       # 融合层的最后一个 Conv2d (out_dim 映射)
        model.tdr.spatial_gate[-1], # Sigmoid 前的 Conv2d (空间门控)
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        # TDR 融合层输出已是 [B, C, H, W]，无需 reshape_transform
    )
    grayscale_cam = cam(
        input_tensor=input_tensor,
        eigen_smooth=True,
        aug_smooth=True,
    )
    grayscale_cam = grayscale_cam[0, :]

    orig_h, orig_w = rgb_img_float.shape[:2]
    heatmap = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    sigma = max(orig_w, orig_h) / 80.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    visualization = show_cam_on_image(rgb_img_float, heatmap, use_rgb=True)
    return visualization, heatmap


def generate_detail_map(heatmap_no_tdr, heatmap_tdr, rgb_img_float):
    """
    生成细节差异图
    蓝色: TDR 减少关注的区域
    白色: 关注度相同的区域
    红色: TDR 额外关注的区域 → 即 TDR 恢复的面部纹理细节
    """
    orig_h, orig_w = rgb_img_float.shape[:2]

    # 归一化到 [0, 1]
    h1 = (heatmap_no_tdr - heatmap_no_tdr.min()) / (heatmap_no_tdr.max() - heatmap_no_tdr.min() + 1e-8)
    h2 = (heatmap_tdr - heatmap_tdr.min()) / (heatmap_tdr.max() - heatmap_tdr.min() + 1e-8)

    # 差异 = TDR - Baseline (range: [-1, 1])
    diff = h2 - h1

    # 映射到蓝-白-红色谱
    # 负值 → 蓝色 (TDR 关注更少)，正值 → 红色 (TDR 关注更多)
    diff_vis = np.zeros((orig_h, orig_w, 3), dtype=np.float32)

    # 红色通道: 正值
    diff_vis[:, :, 0] = np.clip(diff, 0, 1)
    # 蓝色通道: 负值 (取反)
    diff_vis[:, :, 2] = np.clip(-diff, 0, 1)
    # 绿色通道: 1 - |diff| (越接近 0 越白)
    diff_vis[:, :, 1] = 1.0 - np.abs(diff)
    # 亮度由原图提供基础
    diff_vis[:, :, :] *= 0.7

    # 叠加到原图上（30% 原图 + 70% 差异热力）
    alpha = 0.6
    result = rgb_img_float * (1 - alpha) + diff_vis * alpha
    result = np.clip(result, 0, 1)

    return result


def predict_emotion(model, input_tensor, device):
    """预测表情并打印"""
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
        label = EMOTIONS[pred.item()]
    return label, conf.item()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TDR 对比热力图生成")
    parser.add_argument('image', nargs='?', type=str, default=None, help='输入图片路径')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--swin-model', type=str, default=SWIN_MODEL_PATH, help='Swin 权重路径')
    parser.add_argument('--tdr-model', type=str, default=TDR_MODEL_PATH, help='Swin+TDR 权重路径')
    args = parser.parse_args()

    # ── 获取图片路径 ──
    if args.image:
        image_path = args.image
    else:
        image_path = input("请输入图片路径: ").strip().strip("'\"")

    # ── 读取图片 ──
    rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb_img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]
    rgb_img_float = rgb_img / 255.0
    print(f"📷 图片尺寸: {orig_w}×{orig_h}")

    # ── 预处理（与训练时一致的 ImageNet 归一化） ──
    rgb_img_resized = cv2.resize(rgb_img_float, (224, 224))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = (rgb_img_resized - mean) / std
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # 1. 无 TDR 热力图（标准 Swin）
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("📌 模型 1: Swin Transformer (无 TDR)")
    print("=" * 50)
    model_swin = load_swin_model(args.swin_model, DEVICE)
    label_swin, conf_swin = predict_emotion(model_swin, input_tensor, DEVICE)
    print(f"   预测: {label_swin} (置信度: {conf_swin:.2%})")
    print("   🔥 生成 Grad-CAM (无 TDR)...")
    vis_no_tdr, heatmap_no_tdr = generate_gradcam_swin(model_swin, input_tensor, rgb_img_float, DEVICE)

    out1 = os.path.join(args.output_dir, f"{base_name}_1_no_tdr.png")
    cv2.imwrite(out1, cv2.cvtColor(vis_no_tdr, cv2.COLOR_RGB2BGR))
    print(f"   ✅ 已保存: {out1}")

    # ═══════════════════════════════════════════════════════
    # 2. 有 TDR 热力图（Swin + TDR）
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("📌 模型 2: Swin Transformer + TDR")
    print("=" * 50)
    model_tdr = load_tdr_model(args.swin_model, args.tdr_model, DEVICE)
    label_tdr, conf_tdr = predict_emotion(model_tdr, input_tensor, DEVICE)
    print(f"   预测: {label_tdr} (置信度: {conf_tdr:.2%})")
    print("   🔥 生成 Grad-CAM (TDR)...")
    vis_tdr, heatmap_tdr = generate_gradcam_tdr(model_tdr, input_tensor, rgb_img_float, DEVICE)

    out2 = os.path.join(args.output_dir, f"{base_name}_2_with_tdr.png")
    cv2.imwrite(out2, cv2.cvtColor(vis_tdr, cv2.COLOR_RGB2BGR))
    print(f"   ✅ 已保存: {out2}")

    # ═══════════════════════════════════════════════════════
    # 3. 细节差异图（TDR 恢复的纹理细节）
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("📌 图 3: 细节差异图 (蓝←相同→红)")
    print("   红色区域 = TDR 额外关注的面部纹理细节")
    print("=" * 50)
    detail_vis = generate_detail_map(heatmap_no_tdr, heatmap_tdr, rgb_img_float)

    out3 = os.path.join(args.output_dir, f"{base_name}_3_detail_diff.png")
    cv2.imwrite(out3, cv2.cvtColor((detail_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"   ✅ 已保存: {out3}")

    # ═══════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("✅ 全部完成！输出文件:")
    print(f"   1️⃣  无 TDR 热力图:  {out1}")
    print(f"   2️⃣  有 TDR 热力图:  {out2}")
    print(f"   3️⃣  细节差异图:    {out3}")
    print(f"\n   预测对比:")
    print(f"      Swin (无TDR): {label_swin} ({conf_swin:.2%})")
    print(f"      Swin+TDR:     {label_tdr} ({conf_tdr:.2%})")
    print("=" * 50)


if __name__ == "__main__":
    main()
