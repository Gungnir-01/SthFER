import torch
import sys
import os
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utilities.vit_model import CustomViTWithHead

# ---------- 配置 ----------
MODEL_PATH = "Models/ViT_Model/best_model_vit.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------- 加载模型 ----------
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

# ---------- 图像预处理 ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- 获取图像路径 ----------
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("请输入图片路径（支持拖拽）: ").strip().strip("'\"")

# ---------- 推理 ----------
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)

print(f"\n✅ ViT 识别结果: {EMOTIONS[pred.item()]} (置信度: {conf.item():.2f})")
print("-" * 40)
for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
    bar = "█" * int(prob.item() * 30)
    print(f"  {emotion:10s}: {prob.item():.4f}  {bar}")
"