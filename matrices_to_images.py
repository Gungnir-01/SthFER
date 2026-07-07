#!/usr/bin/env python
"""
将 yikuyou/ 中的四个热力矩阵 (.npy) 转换为可视化的热力图 PNG 图片
"""
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MATRIX_DIR = "yikuyou"
OUTPUT_DIR = "yikuyou"

STAGE_LABELS = {
    "stage0": "Stage 0 (Texture/Edges)",
    "stage1": "Stage 1 (Local Shapes)",
    "stage2": "Stage 2 (Parts/Patterns)",
    "stage3": "Stage 3 (Global Semantics)",
}


def matrix_to_heatmap_image(matrix, output_path, title=None, colormap=cv2.COLORMAP_JET):
    """
    将热力矩阵转换为可视化的热力图 PNG 图片
    使用 OpenCV 生成彩色热力图 + Matplotlib 生成带 colorbar 的版本
    """
    # 归一化到 [0, 1]
    matrix_norm = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)

    # 方式1: 用 OpenCV 生成干净的彩色热力图
    heatmap_colored = cv2.applyColorMap(
        (matrix_norm * 255).astype(np.uint8), colormap
    )
    cv2.imwrite(output_path, heatmap_colored)


def matrix_to_heatmap_with_colorbar(matrix, output_path, title=None):
    """使用 matplotlib 生成带 colorbar 和标题的热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='jet', aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_four_matrix_comparison(matrices, titles, output_path):
    """创建四个矩阵的并排对比图（带 colorbar）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Swin Transformer — 4-Stage Attention Matrices", fontsize=16, fontweight='bold')

    for ax, matrix, title in zip(axes.flat, matrices, titles):
        im = ax.imshow(matrix, cmap='jet', aspect='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 找到唯一的图片名前缀
    npy_files = sorted([f for f in os.listdir(MATRIX_DIR) if f.endswith('.npy')])
    if not npy_files:
        print("❌ 未找到 .npy 文件")
        return

    # 提取前缀
    prefix = npy_files[0].split('_stage')[0]
    print(f"📊 处理矩阵，前缀: {prefix}")

    matrices = []
    titles = []

    for stage_key in ["stage0", "stage1", "stage2", "stage3"]:
        # 低分辨率原始矩阵 (224×224)
        matrix_file = f"{prefix}_{stage_key}_matrix.npy"
        matrix_path = os.path.join(MATRIX_DIR, matrix_file)
        matrix = np.load(matrix_path)
        matrices.append(matrix)
        titles.append(STAGE_LABELS[stage_key])

        # 生成纯热力图（无 colorbar）
        heatmap_path = os.path.join(OUTPUT_DIR, f"{prefix}_{stage_key}_matrix_heatmap.png")
        matrix_to_heatmap_image(matrix, heatmap_path)

        # 生成带 colorbar 的热力图
        cb_path = os.path.join(OUTPUT_DIR, f"{prefix}_{stage_key}_matrix_colorbar.png")
        matrix_to_heatmap_with_colorbar(matrix, cb_path, title=STAGE_LABELS[stage_key])

        print(f"   ✅ {stage_key}: matrix_heatmap.png + matrix_colorbar.png")

    # 生成四矩阵对比图
    comparison_path = os.path.join(OUTPUT_DIR, f"{prefix}_matrix_comparison.png")
    create_four_matrix_comparison(matrices, titles, comparison_path)
    print(f"   ✅ 四矩阵对比图: matrix_comparison.png")

    print(f"\n🏁 全部完成! 结果保存在: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png') and ('matrix_heatmap' in f or 'matrix_colorbar' in f or 'matrix_comparison' in f):
            fsize = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"     📄 {f} ({fsize:.1f} KB)")


if __name__ == "__main__":
    main()
