"""
ViT Attention Heatmap Generator
对测试图片生成 ViT 注意力热力图，展示模型关注的区域
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.vit_model import CustomViTWithHead


class ViTAttentionExtractor:
    """
    从 vit_pytorch 的 ViT 模型中提取 attention maps
    """
    def __init__(self, model):
        self.model = model
        self.attentions = []  # 存储每层的 attention weights
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """在 transformer 的每一层 Attention 上注册 hook"""
        # model.vit.transformer.layers 是 ModuleList
        # 每个元素是 [Attention, FeedForward]
        for i, (attn_module, _) in enumerate(self.model.vit.transformer.layers):
            # hook 在 softmax 之后 (self.attend)
            handle = attn_module.attend.register_forward_hook(
                self._create_hook_fn(i)
            )
            self.handles.append(handle)

    def _create_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):
            # output 是 attention weights, shape: [B, H, N, N]
            # B=batch, H=heads, N=seq_len (patches+CLS)
            self.attentions[layer_idx] = output.detach().cpu()
        return hook_fn

    def extract(self, img_tensor):
        """
        前向传播并提取所有层的 attention
        img_tensor: [1, 3, 224, 224]
        返回: list of [1, H, N, N] 每层一个
        """
        self.attentions = [None] * len(self.model.vit.transformer.layers)
        with torch.no_grad():
            self.model(img_tensor)
        return self.attentions

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attention_to_heatmap(attn_weights, patch_size=32, img_size=224):
    """
    将 CLS token 对所有 patch 的 attention 转为热力图

    参数:
        attn_weights: [H, N, N] — 单张图所有 head 的 attention
        patch_size: 每个 patch 的像素大小
        img_size: 原始图像大小

    返回:
        heatmap: [img_size, img_size] — 归一化后的热力图 (numpy)
    """
    num_heads = attn_weights.shape[0]
    # 取 CLS token (第0个) 对所有 patch (1: 之后) 的 attention
    # 对所有 head 取平均
    cls_attn = attn_weights[:, 0, 1:]  # [H, num_patches]
    cls_attn = cls_attn.mean(dim=0)    # [num_patches]

    num_patches = cls_attn.shape[0]
    grid_size = int(num_patches ** 0.5)
    cls_attn = cls_attn.reshape(grid_size, grid_size)

    # 上采样到原始图像大小
    cls_attn = cls_attn.numpy()
    heatmap = cv2.resize(cls_attn, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # 归一化到 [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原图上

    参数:
        pil_img: PIL Image
        heatmap: [H, W] numpy array, 值在 [0, 1]
        alpha: 叠加透明度
        colormap: OpenCV colormap

    返回:
        overlay: PIL Image
    """
    # PIL → numpy (RGB)
    img_np = np.array(pil_img.convert('RGB'))
    img_np = cv2.resize(img_np, (heatmap.shape[1], heatmap.shape[0]))

    # 热力图转为彩色
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), colormap
    )  # BGR
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 叠加
    overlay = (img_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def create_side_by_side(original, overlay):
    """创建并排对比图"""
    w, h = original.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(original, (0, 0))
    combined.paste(overlay, (w, 0))
    return combined


def main():
    # ---------- 配置 ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "Models", "ViT_Model", "best_model_vit.pth")

    # 测试图片目录
    input_dir = r"C:\Users\Jesse\Desktop\materials"

    # 输出目录
    output_dir = os.path.join(project_root, "heatmap_results")
    os.makedirs(output_dir, exist_ok=True)

    # 类别名称
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # 图像预处理（与训练时验证集一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 反归一化（用于显示原图）
    denorm = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    # ---------- 加载模型 ----------
    print("Loading model...")
    model = CustomViTWithHead(
        image_size=224,
        patch_size=32,
        num_classes=7,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")

    # ---------- 初始化 attention 提取器 ----------
    extractor = ViTAttentionExtractor(model)

    # ---------- 处理每张测试图片 ----------
    for i in range(1, 22):
        filename = f"{i}.png"
        filepath = os.path.join(input_dir, filename)

        if not os.path.exists(filepath):
            print(f"  [!] {filename} not found, skipping...")
            continue

        print(f"Processing {filename}...")

        # 加载图像
        pil_img = Image.open(filepath).convert('RGB')
        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # 提取 attention
        attentions = extractor.extract(img_tensor)

        # 获取预测结果
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_label = class_names[pred_idx]
            confidence = probs[pred_idx] * 100

        # ---------- 生成热力图 ----------
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"{filename} — Predicted: {pred_label} ({confidence:.1f}%)", fontsize=16)

        # 子图1: 原图
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # 子图2-7: 每层的 attention heatmap (共6层)
        for layer_idx in range(6):
            row = (layer_idx + 1) // 4
            col = (layer_idx + 1) % 4
            ax = axes[row, col]

            attn = attentions[layer_idx]  # [1, H, N, N]
            if attn is None:
                ax.set_title(f"Layer {layer_idx + 1} — No Data")
                ax.axis('off')
                continue

            heatmap = attention_to_heatmap(attn[0], patch_size=32, img_size=224)
            overlay_img = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.6)
            ax.imshow(overlay_img)
            ax.set_title(f"Layer {layer_idx + 1} Attention")
            ax.axis('off')

        # 子图8: 概率分布
        colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#C9B1FF', '#FF8C42', '#A0A0A0']
        bars = axes[1, 3].barh(class_names, probs * 100, color=colors)
        axes[1, 3].set_xlim(0, 100)
        axes[1, 3].set_xlabel("Probability (%)")
        axes[1, 3].set_title("Class Probabilities")
        for bar, p in zip(bars, probs):
            axes[1, 3].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{p*100:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{i}_heatmap.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved: {fig_path}")

    # ---------- 清理 ----------
    extractor.remove_hooks()
    print(f"\nAll done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()