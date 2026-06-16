import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.vit_model import CustomViT, CustomViTWithHead


def evaluate_vit(model_path, data_dir, device, model_type='vit_with_head'):
    """
    评估 ViT 模型
    model_type: 'vit' 或 'vit_with_head'
    """
    # 构建模型
    if model_type == 'vit_with_head':
        model = CustomViTWithHead(image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048)
    else:
        model = CustomViT(image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048)

    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 输出分类报告
    print("=" * 60)
    print("ViT 模型评估 - 分类报告")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:")
    print(cm)

    # 计算总体准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"总体准确率: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    model_path = "Models/ViT_Model/best_model_vit.pth"
    data_dir = "FER2013_processed/test"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_vit(model_path, data_dir, device, model_type='vit_with_head')
