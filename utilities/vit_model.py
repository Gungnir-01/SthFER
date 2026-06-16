import torch
import torch.nn as nn
from vit_pytorch import ViT

class CustomViT(nn.Module):
    """
    ViT 面部表情识别模型，使用 vit-pytorch 库构建
    输入: 224x224 RGB 图像
    输出: 7 类情感 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    """
    def __init__(self, image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super(CustomViT, self).__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, x):
        return self.vit(x)

# 可选：一个加了自定义分类头的版本，与 Swin Transformer 风格更一致
class CustomViTWithHead(nn.Module):
    """
    带自定义分类头的 ViT，结构更接近项目中的 Swin Transformer
    """
    def __init__(self, image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super(CustomViTWithHead, self).__init__()
        # ViT backbone (不包含分类头)
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0,  # 移除默认分类头，手动添加
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 快速测试
    model = CustomViT(image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 预期: (2, 7)
    print(f"模型可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试带自定义头的版本
    model2 = CustomViTWithHead(image_size=224, patch_size=32, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048)
    output2 = model2(dummy_input)
    print(f"带自定义头输出形状: {output2.shape}")  # 预期: (2, 7)