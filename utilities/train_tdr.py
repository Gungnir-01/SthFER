#!/usr/bin/env python
"""
Swin+TDR 模型训练脚本
策略: 冻结 Swin backbone → 训练 TDR + 分类头 → 解冻微调
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.tdr import SwinTDR


def train_one_epoch(model, dataloader, optimizer, criterion, device, desc="Train"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=desc)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Val"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def set_trainable(model, trainable_modules):
    """只解冻指定模块，其余冻结"""
    for name, param in model.named_parameters():
        param.requires_grad = any(m in name for m in trainable_modules)


def main():
    parser = argparse.ArgumentParser(description="Swin+TDR 训练")
    parser.add_argument('--epochs', type=int, default=30, help='总训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='TDR 初始学习率')
    parser.add_argument('--lr_finetune', type=float, default=1e-5, help='微调学习率')
    parser.add_argument('--unfreeze_epoch', type=int, default=15, help='从第几轮解冻 backbone')
    parser.add_argument('--data_dir', type=str, default='FER2013_processed')
    parser.add_argument('--swin_weights', type=str, default='Models/Swin_Transformer/best_model.pth')
    parser.add_argument('--save_path', type=str, default='Models/Swin_Transformer/best_model_tdr.pth')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {DEVICE}")

    # ── 数据增强 ──
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"📦 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 类别: {train_dataset.classes}")

    # ── 构建模型 ──
    model = SwinTDR(
        model_name='swin_small_patch4_window7_224',
        num_classes=7,
        pretrained_backbone=False,
        pretrained_swin_path=args.swin_weights,   # 从标准 Swin checkpoint 迁移
        tdr_hidden_dim=256,
        dropout_rate=0.5,
    )

    model.to(DEVICE)

    # ── 阶段一: 冻结 backbone，只训练 TDR + 分类头 ──
    print(f"\n🔒 阶段一 (第 1-{args.unfreeze_epoch} 轮): 冻结 backbone，训练 TDR + 分类头")
    set_trainable(model, ['tdr', 'classifier'])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.unfreeze_epoch)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # 解冻 backbone
        if epoch == args.unfreeze_epoch + 1:
            print(f"\n🔓 阶段二 (第 {epoch}-{args.epochs} 轮): 解冻 backbone，全模型微调")
            set_trainable(model, ['backbone', 'tdr', 'classifier'])
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.unfreeze_epoch
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, f"Epoch {epoch}/{args.epochs}")
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"   Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"   ✅ 保存最佳模型 → {args.save_path} (Val Acc: {val_acc:.4f})")

    print(f"\n🏆 训练完成! 最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
