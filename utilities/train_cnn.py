# utilities/train_cnn.py
"""CNN 模型训练脚本 —— 用于 FER2013 面部表情识别（7 分类）"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
import argparse


# ═══════════════════════════════════════════════════════════════
# CNN 模型定义
# ═══════════════════════════════════════════════════════════════

class FER_CNN(nn.Module):
    """
    面部表情识别 CNN 模型
    - 4 个卷积块，逐层提取高阶语义特征
    - 最后一个卷积层 (block4_conv2) 作为 Grad-CAM 目标层
    - 输入: 224×224×3  RGB 人脸图片
    - 输出: 7 类表情 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    """
    def __init__(self, num_classes=7, dropout_rate=0.4):
        super(FER_CNN, self).__init__()

        # ---- Block 1: 224 → 112 ----
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_bn1   = nn.BatchNorm2d(64)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_bn2   = nn.BatchNorm2d(64)
        self.block1_pool  = nn.MaxPool2d(2, 2)
        self.block1_drop  = nn.Dropout2d(0.25)

        # ---- Block 2: 112 → 56 ----
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_bn1   = nn.BatchNorm2d(128)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_bn2   = nn.BatchNorm2d(128)
        self.block2_pool  = nn.MaxPool2d(2, 2)
        self.block2_drop  = nn.Dropout2d(0.25)

        # ---- Block 3: 56 → 28 ----
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_bn1   = nn.BatchNorm2d(256)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_bn2   = nn.BatchNorm2d(256)
        self.block3_pool  = nn.MaxPool2d(2, 2)
        self.block3_drop  = nn.Dropout2d(0.3)

        # ---- Block 4: 28 → 14（Grad-CAM 目标层所在） ----
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_bn1   = nn.BatchNorm2d(512)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # ← Grad-CAM 目标
        self.block4_bn2   = nn.BatchNorm2d(512)
        self.block4_pool  = nn.MaxPool2d(2, 2)
        self.block4_drop  = nn.Dropout2d(0.3)

        # ---- 分类器 ----
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))   # 14→4
        self.flatten     = nn.Flatten()
        self.fc1         = nn.Linear(512 * 4 * 4, 256)
        self.fc1_bn      = nn.BatchNorm1d(256)
        self.fc1_drop    = nn.Dropout(dropout_rate)
        self.fc2         = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.block1_bn1(self.block1_conv1(x)))
        x = F.relu(self.block1_bn2(self.block1_conv2(x)))
        x = self.block1_drop(self.block1_pool(x))

        # Block 2
        x = F.relu(self.block2_bn1(self.block2_conv1(x)))
        x = F.relu(self.block2_bn2(self.block2_conv2(x)))
        x = self.block2_drop(self.block2_pool(x))

        # Block 3
        x = F.relu(self.block3_bn1(self.block3_conv1(x)))
        x = F.relu(self.block3_bn2(self.block3_conv2(x)))
        x = self.block3_drop(self.block3_pool(x))

        # Block 4
        x = F.relu(self.block4_bn1(self.block4_conv1(x)))
        x = F.relu(self.block4_bn2(self.block4_conv2(x)))
        x = self.block4_drop(self.block4_pool(x))

        # Classifier
        x = self.global_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════
# 训练函数
# ═══════════════════════════════════════════════════════════════

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


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 CNN 面部表情识别模型")
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--data_dir', type=str, default='FER2013_processed', help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='Models/CNN', help='模型保存目录')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout 概率')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    # ---- 数据增强 & 预处理 ----
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- 加载数据 ----
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = os.path.join(args.data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"错误: 训练目录不存在: {train_dir}")
        print("请先运行: python utilities/preprocess_data.py")
        exit(1)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform) if os.path.exists(val_dir) else None
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True) if val_dataset else None

    print(f"训练集: {len(train_dataset)} 张图片, 类别: {train_dataset.classes}")
    if val_dataset:
        print(f"验证集: {len(val_dataset)} 张图片")

    # ---- 构建模型 ----
    model = FER_CNN(num_classes=7, dropout_rate=args.dropout).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    # ---- 训练循环 ----
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            current_acc = val_acc
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            current_acc = train_acc

        scheduler.step(current_acc)

        if current_acc > best_acc:
            best_acc = current_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'accuracy': best_acc,
                'model_args': {'num_classes': 7, 'dropout_rate': args.dropout},
            }, save_path)
            print(f"  => 保存最佳模型到 {save_path} (acc: {best_acc:.4f})")

    print(f"\n训练完成! 最佳准确率: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")
