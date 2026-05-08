import torch
import timm
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

def evaluate_model(model_path, data_dir, transform, device):
    # 构建 Swin-Small 模型（注意：这里必须是 small，与作者权重匹配）
    model = timm.create_model('swin_small_patch4_window7_224', num_classes=7)
    
    # 加载权重，允许非严格匹配
    state_dict = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # 输出缺失/多余键以供检查
    if missing_keys:
        print(f"警告：有 {len(missing_keys)} 个键缺失（通常无影响）: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"注意：有 {len(unexpected_keys)} 个多余键（已忽略）")
    
    model.to(device)
    model.eval()

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    evaluate_model("Models/Swin_Transformer/best_model.pth", "FER2013_processed/test", transform, 'cuda')