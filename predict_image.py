import torch
import timm
from torchvision import transforms
from PIL import Image
import sys

# ---------- 配置 ----------
MODEL_PATH = "Models/Swin_Transformer/best_model.pth"   # 作者模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------- 加载模型 ----------
model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)   # 忽略缺失的非关键缓冲
model.to(DEVICE)
model.eval()

# ---------- 图片预处理 ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- 读取图片路径 ----------
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("请输入图片路径（拖拽图片到这里也行）: ").strip().strip("'\"")  # 支持拖入

# ---------- 推理 ----------
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, dim=0)

print(f"\n✅ 识别结果: {EMOTIONS[pred.item()]} (置信度: {conf.item():.2f})")
for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
    print(f"  {emotion}: {prob.item():.4f}")