# utilities/train_model.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm



# Custom model class
class CustomSwinTransformer(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(CustomSwinTransformer, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--data_dir', type=str, default='FER2013_processed', help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='Models/Swin_Transformer', help='模型保存目录')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"错误: 训练目录不存在: {train_dir}")
        print("请先运行: python utilities/preprocess_data.py")
        exit(1)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"训练集大小: {len(train_dataset)} 张图片, 类别: {train_dataset.classes}")

    model = CustomSwinTransformer(pretrained=True, num_classes=7).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch + 1}/{args.epochs}: Loss={loss:.4f}, Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  => 保存最佳模型到 {save_path} (acc: {acc:.4f})")

    print(f"\n训练完成! 最佳准确率: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")