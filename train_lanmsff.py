"""
LANMSFF 模型训练脚本
Lightweight Attention Network with Multi-Scale Feature Fusion for FER

模型来源: external/LANMSFF/model.py
数据集: FER2013_processed (7 分类, 64×64 灰度图)

用法:
    python train_lanmsff.py
    python train_lanmsff.py --epochs 100 --batch_size 32 --lr 0.001
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
import sys
import argparse

# 添加 external/LANMSFF 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external', 'LANMSFF'))
from model import LANMSFF


# ═══════════════════════════════════════════════════════════════
# 训练 / 验证函数
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


def test_model(model, dataloader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = correct.double() / total
    return test_loss, test_acc.item()


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 LANMSFF 面部表情识别模型")
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--data_dir', type=str, default='FER2013_processed', help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='Models/LANMSFF', help='模型保存目录')
    parser.add_argument('--num_classes', type=int, default=7, help='表情类别数')
    parser.add_argument('--patience', type=int, default=14, help='早停耐心值')
    parser.add_argument('--lr_patience', type=int, default=9, help='学习率衰减耐心值')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    # ---- 数据增强 & 预处理 ----
    # LANMSFF 输入: 64×64 灰度图 (1 channel)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # ---- 加载数据 ----
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    if not os.path.exists(train_dir):
        print(f"错误: 训练目录不存在: {train_dir}")
        print("请先运行: python utilities/preprocess_data.py")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform) if os.path.exists(val_dir) else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True) if val_dataset else None

    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform) if os.path.exists(test_dir) else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True) if test_dataset else None

    print(f"训练集: {len(train_dataset)} 张图片, 类别: {train_dataset.classes}")
    if val_dataset:
        print(f"验证集: {len(val_dataset)} 张图片")
    if test_dataset:
        print(f"测试集: {len(test_dataset)} 张图片")

    # ---- 构建模型 ----
    model = LANMSFF(num_classes=args.num_classes, input_channels=1).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=args.weight_decay
    )

    # ReduceLROnPlateau: 监控 val_loss, factor=0.1, patience=9
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=args.lr_patience
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    # ---- 训练循环 ----
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查 + 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'model_args': {
                        'num_classes': args.num_classes,
                        'input_channels': 1,
                    },
                }, save_path)
                print(f"  => 保存最佳模型到 {save_path} (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  => 验证损失未改善 ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n早停触发! 在第 {epoch+1} 轮停止训练")
                break
        else:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")

    # ---- 加载最佳模型并进行测试 ----
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"最佳模型来自 Epoch {checkpoint['epoch']}, "
              f"Val Loss: {checkpoint['val_loss']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}")

        if test_loader:
            print("\n在测试集上评估...")
            test_loss, test_acc = test_model(model, test_loader, criterion, DEVICE)
            print(f"测试集结果: Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    print(f"\n模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")
