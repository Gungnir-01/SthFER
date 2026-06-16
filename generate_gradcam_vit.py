import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utilities.vit_model import CustomViTWithHead
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# ---------- 配置 ----------
MODEL_PATH = "Models/ViT_Model/best_model_vit.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------- 加载 ViT 模型 ----------
model = CustomViTWithHead(
    image_size=224,
    patch_size=32,
    num_classes=7,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ---------- Grad-CAM 包装器（提取 CLS token） ----------
class ViTGradCAMWrapper(nn.Module):
    """包装 ViT 模型，使输出仅为 CLS token 的 7 类 logits，兼容 Grad-CAM"""
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, x):
        out = self.vit_model(x)      # [B, 50, 7]
        return out[:, 0, :]           # 仅取 CLS token [B, 7]

wrapped_model = ViTGradCAMWrapper(model)

# ---------- Grad-CAM 目标层 ----------
# ViT 最后一层 attention 的 to_qkv（QKV 联合投影），
# 此处 patch token 梯度通过 attention 机制有效流动，热力权重最明显
target_layers = [model.vit.transformer.layers[-1][0].to_qkv]


def vit_reshape_transform(tensor, height=14, width=14):
    """
    将 ViT 的 (B, L, C) reshape 为 (B, C, H, W)
    ViT patch_size=32, image_size=224 → 7x7 patches
    但这里 dim=1024, depth=6, 输出的是 (B, num_patches+1, dim)
    """
    if tensor.dim() == 4:
        return tensor
    elif tensor.dim() == 3:
        B, L, C = tensor.shape
        # 去掉 CLS token
        L = L - 1
        import math
        hw = int(math.sqrt(L))
        if hw * hw != L:
            raise ValueError(f"序列长度 {L} 不是完全平方数")
        result = tensor[:, 1:, :].reshape(B, hw, hw, C).permute(0, 3, 1, 2)
        return result
    return tensor


# ---------- 读取图像 ----------
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("请输入图片路径: ").strip().strip("'\"")

rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if rgb_img is None:
    raise FileNotFoundError(f"无法读取图片: {image_path}")
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

orig_h, orig_w = rgb_img.shape[:2]
rgb_img_resized = cv2.resize(rgb_img, (224, 224))

# 预处理
input_tensor = torch.from_numpy(rgb_img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

# ---------- 预测 ----------
with torch.no_grad():
    outputs = wrapped_model(input_tensor)    # [1, 7]
    _, pred = torch.max(outputs, 1)
    predicted_label = EMOTIONS[pred.item()]
    print(f"预测表情: {predicted_label}")

# ---------- Grad-CAM ----------
cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers,
    reshape_transform=vit_reshape_transform
)

grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True, aug_smooth=True)
grayscale_cam = grayscale_cam[0, :]

# 放大到原图尺寸并平滑
heatmap_resized = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
sigma = max(orig_w, orig_h) / 80.0
heatmap_resized = gaussian_filter(heatmap_resized, sigma=sigma)

# 叠加
rgb_img_float = rgb_img / 255.0
visualization = show_cam_on_image(rgb_img_float, heatmap_resized, use_rgb=True)

# 保存
output_dir = "heatmaps_vit"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"gradcam_vit_{os.path.basename(image_path)}"
output_path = os.path.join(output_dir, output_filename)
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ ViT 热力图已保存至: {output_path}")
