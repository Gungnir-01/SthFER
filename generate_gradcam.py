import torch
import timm
import cv2
import numpy as np
import os
import sys
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------- 配置 ----------
MODEL_PATH = "Models/Swin_Transformer/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------- 加载模型 ----------
model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

def swin_transformer_reshape_transform(tensor, height=7, width=7):
    # 如果已经是 4 维 (B, C, H, W)，直接返回
    if tensor.dim() == 4:
        return tensor
    # 如果是 3 维 (B, L, C)，进行重塑
    elif tensor.dim() == 3:
        B, L, C = tensor.shape
        # 自动计算空间尺寸
        HW = int(math.sqrt(L))
        if HW * HW != L:
            raise ValueError(f"序列长度 {L} 不是完全平方数，无法推断空间尺寸")
        result = tensor.reshape(B, HW, HW, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return result
    else:
        return tensor

# ---------- 目标层 ----------
target_layers = [model.layers[-1].blocks[-1].norm1]

# ---------- 读取图片（支持中文） ----------
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("请输入图片路径: ").strip().strip("'\"")

rgb_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if rgb_img is None:
    raise FileNotFoundError(f"无法读取图片: {image_path}")
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

# 保留原图尺寸
orig_h, orig_w = rgb_img.shape[:2]
# 模型需要 224x224
rgb_img_resized = cv2.resize(rgb_img, (224, 224))

# 预处理为模型输入
input_tensor = torch.from_numpy(rgb_img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

# ---------- 预测表情 ----------
with torch.no_grad():
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, 1)
    predicted_label = EMOTIONS[pred.item()]
    print(f"预测表情: {predicted_label}")

# ---------- 生成 Grad-CAM ----------
cam = GradCAM(model=model,
              target_layers=target_layers,
              reshape_transform=swin_transformer_reshape_transform)

# eigen_smooth=True 可以消除条纹噪声
grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)
grayscale_cam = grayscale_cam[0, :]   # 取出 (H, W)

# 将 7x7 的热力图平滑放大到原图尺寸
heatmap_resized = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

# 叠加原图与热力图
rgb_img_float = rgb_img / 255.0
visualization = show_cam_on_image(rgb_img_float, heatmap_resized, use_rgb=True)

# ---------- 保存 ----------
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"gradcam_{os.path.basename(image_path)}"
output_path = os.path.join(output_dir, output_filename)
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ 热力图已保存至: {output_path}")