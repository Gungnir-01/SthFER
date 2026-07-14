import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.vit_model import CustomViT, CustomViTWithHead


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)                # [B, N, 7]
        cls_outputs = outputs[:, 0, :]         # 取 CLS token [B, 7]
        loss = criterion(cls_outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(cls_outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)                # [B, N, 7]
            cls_outputs = outputs[:, 0, :]         # 取 CLS token [B, 7]
            loss = criterion(cls_outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(cls_outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


if __name__ == "__main__":
    # ---------- 配置 ----------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "FER2013_processed")  # 预处理后的数据集路径
    num_epochs = 50
    batch_size = 32
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_path = os.path.join(project_root, "Models", "ViT_Model", "best_model_vit.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # ---------- 数据增强与预处理 ----------
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------- 加载数据集 ----------
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # ---------- 模型、损失函数、优化器 ----------
    # 使用带自定义头的版本（与 Swin 风格一致），也可以换成 CustomViT
    model = CustomViTWithHead(
        image_size=224,
        patch_size=32,
        num_classes=7,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---------- 训练循环 ----------
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"*** 新的最佳模型已保存 (Val Acc: {val_acc:.4f}) ***")

    print(f"\n训练完成。最佳验证准确率: {best_val_acc:.4f}")

