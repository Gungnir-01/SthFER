"""
TDR (Token Detail Recovery) 模块
用于 Swin Transformer 的多尺度特征融合，恢复面部纹理细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class TDRModule(nn.Module):
    """
    Token Detail Recovery Module
    融合 Swin 四个 Stage 的多尺度特征，通过注意力门控机制恢复空间细节

    输入: 4 个 Stage 的特征图 (NHWC 格式，来自 timm features_only)
      - Stage 0: [B, 56, 56, 96]   细粒度纹理
      - Stage 1: [B, 28, 28, 192]  中等粒度
      - Stage 2: [B, 14, 14, 384]  粗粒度
      - Stage 3: [B, 7, 7, 768]    强语义 (最终分类依赖)
    输出: [B, 768, 7, 7]  融合后的增强特征
    """

    def __init__(self, in_dims=(96, 192, 384, 768), hidden_dim=256, out_dim=768):
        super().__init__()

        # ── 通道投影：各阶段 → hidden_dim ──
        self.proj_s0 = nn.Sequential(
            nn.Conv2d(in_dims[0], hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_s1 = nn.Sequential(
            nn.Conv2d(in_dims[1], hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_s2 = nn.Sequential(
            nn.Conv2d(in_dims[2], hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_s3 = nn.Sequential(
            nn.Conv2d(in_dims[3], hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── 注意力门控（Spatial Gate）：学习哪些空间位置需要细节恢复 ──
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # ── 融合卷积 ──
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # ── 细节残差增强 ──
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim // 16),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """
        feats: list of [f0, f1, f2, f3]
               每个是 NHWC 格式 → 需要 permute 到 NCHW
        """
        # NHWC → NCHW
        f0 = feats[0].permute(0, 3, 1, 2).contiguous()  # [B, 96, 56, 56]
        f1 = feats[1].permute(0, 3, 1, 2).contiguous()  # [B, 192, 28, 28]
        f2 = feats[2].permute(0, 3, 1, 2).contiguous()  # [B, 384, 14, 14]
        f3 = feats[3].permute(0, 3, 1, 2).contiguous()  # [B, 768, 7, 7]

        target_size = f3.shape[2:]  # (7, 7)

        # 投影到统一通道
        p0 = self.proj_s0(f0)                                       # [B, 256, 56, 56]
        p1 = self.proj_s1(f1)                                       # [B, 256, 28, 28]
        p2 = self.proj_s2(f2)                                       # [B, 256, 14, 14]
        p3 = self.proj_s3(f3)                                       # [B, 256, 7, 7]

        # 上采样到目标尺寸
        p0 = F.interpolate(p0, size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)

        # 拼接多尺度特征
        concat = torch.cat([p0, p1, p2, p3], dim=1)               # [B, 1024, 7, 7]

        # 空间注意力门控
        gate = self.spatial_gate(concat)                            # [B, 1, 7, 7]

        # 融合
        fused = self.fusion(concat)                                 # [B, 768, 7, 7]

        # 细节增强（门控残差）
        detail = self.detail_enhance(fused)                         # [B, 768, 7, 7]
        out = fused + gate * detail                                 # 门控残差连接

        return out, gate  # 返回融合特征 + 注意力门（用于可视化）


class SwinTDR(nn.Module):
    """
    Swin Transformer + TDR 模型

    架构:
      Swin Backbone (features_only) → TDR 多尺度融合 → 分类头
    """

    def __init__(
        self,
        model_name='swin_small_patch4_window7_224',
        num_classes=7,
        pretrained_backbone=True,
        pretrained_swin_path=None,
        tdr_hidden_dim=256,
        dropout_rate=0.5,
    ):
        super().__init__()

        # ── Swin Backbone (提取多尺度特征) ──
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            num_classes=0,
        )

        # 获取各阶段输出通道数
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
        in_dims = tuple(f.shape[-1] for f in feats)

        # ── 从标准 Swin checkpoint 迁移 backbone 权重 ──
        if pretrained_swin_path and os.path.exists(pretrained_swin_path):
            self._load_swin_weights(pretrained_swin_path)

        # ── TDR 模块 ──
        self.tdr = TDRModule(
            in_dims=in_dims,
            hidden_dim=tdr_hidden_dim,
            out_dim=in_dims[-1],
        )

        # ── 分类头 ──
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self.in_dims = in_dims

    def _load_swin_weights(self, swin_path):
        """
        从标准 Swin checkpoint (num_classes=7) 迁移 backbone 权重
        处理 features_only 的 key 差异：
          layers.i.xxx → layers_i.xxx
          跳过 norm.* 和 head.*
        """
        print(f"📥 迁移 Swin backbone 权重: {swin_path}")
        state_dict = torch.load(swin_path, map_location='cpu')

        new_state = {}
        skipped = 0
        for k, v in state_dict.items():
            # 跳过分类头和最后的 norm
            if k.startswith('head.') or k.startswith('norm.'):
                skipped += 1
                continue
            # layers.i.xxx → layers_i.xxx
            if k.startswith('layers.'):
                parts = k.split('.', 2)  # ['layers', '0', 'blocks.0.norm1.weight']
                new_k = f"{parts[0]}_{parts[1]}.{parts[2]}"
            else:
                new_k = k
            new_state[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
        loaded = len(new_state)
        print(f"   ✅ {loaded} 个 backbone 参数已迁移 (跳过 {skipped} 个 norm/head)")

    def forward(self, x):
        feats = self.backbone(x)
        fused, gate = self.tdr(feats)
        out = self.classifier(fused)
        return out

    def forward_with_gate(self, x):
        """返回分类结果 + 门控注意力图"""
        feats = self.backbone(x)
        fused, gate = self.tdr(feats)
        out = self.classifier(fused)
        return out, gate


# timm 需要在此导入（避免循环依赖）
import timm
