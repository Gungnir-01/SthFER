"""
LANMSFF Model - PyTorch Implementation
Converted from TensorFlow/Keras notebooks:
  - model.ipynb (main model architecture)
  - MassAtt_module.ipynb (Mixed Attention module)
  - PWFS_module.ipynb (Pixel-Wise Feature Selection module)

Original repo: https://github.com/AE-1129/LANMSFF.git
Paper: Lightweight Attention Network with Multi-Scale Feature Fusion for FER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# Channel Shuffle (from ShuffleNet)
# ===========================
class ChannelShuffle(nn.Module):
    """Channel shuffle operation, splits channels into groups and interleaves them."""
    def __init__(self, groups: int = 2):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        channels_per_group = C // self.groups
        # (B, C, H, W) -> (B, groups, channels_per_group, H, W)
        x = x.view(B, self.groups, channels_per_group, H, W)
        # Transpose group and channel dims -> (B, channels_per_group, groups, H, W)
        x = x.transpose(1, 2).contiguous()
        # Flatten back to (B, C, H, W)
        x = x.view(B, C, H, W)
        return x


# ===========================
# Two-Path Convolution Block
# ===========================
class TwoPathConv(nn.Module):
    """
    Dual-path convolution block:
      - H-path: regular 3×3 conv → SeparableConv → regular 3×3 conv
      - L-path: dilated 3×3 conv → SeparableConv (dilated) → dilated 3×3 conv
    Paths are interleaved via channel shuffle before splitting.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 2):
        super(TwoPathConv, self).__init__()
        filters_per_group = out_channels // 2
        half_in = in_channels // 2  # Each group gets half the channels after chunk

        # Stage 1: each path takes half the input channels
        # H-path (regular convolution)
        self.conv_h1 = nn.Conv2d(half_in, filters_per_group, kernel_size,
                                 stride=stride, padding=padding, bias=False)
        self.bn_h1 = nn.BatchNorm2d(filters_per_group)

        # L-path (dilated convolution)
        self.conv_l1 = nn.Conv2d(half_in, filters_per_group, kernel_size,
                                 stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn_l1 = nn.BatchNorm2d(filters_per_group)

        # Stage 2: SeparableConv (depthwise + pointwise), input = full out_channels
        # H-path depthwise
        self.dw_h2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, groups=out_channels, bias=False)
        self.pw_h2 = nn.Conv2d(out_channels, filters_per_group, 1, bias=False)
        self.bn_h2 = nn.BatchNorm2d(filters_per_group)

        # L-path depthwise (dilated)
        self.dw_l2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=dilation, dilation=dilation,
                               groups=out_channels, bias=False)
        self.pw_l2 = nn.Conv2d(out_channels, filters_per_group, 1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(filters_per_group)

        # Stage 3: regular conv, input = full out_channels
        self.conv_h3 = nn.Conv2d(out_channels, filters_per_group, kernel_size,
                                 stride=stride, padding=padding, bias=False)
        self.bn_h3 = nn.BatchNorm2d(filters_per_group)

        self.conv_l3 = nn.Conv2d(out_channels, filters_per_group, kernel_size,
                                 stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn_l3 = nn.BatchNorm2d(filters_per_group)

        self.shuffle = ChannelShuffle(groups=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel shuffle then split into two groups
        x = self.shuffle(x)
        group1, group2 = torch.chunk(x, 2, dim=1)

        # Stage 1
        h1 = self.relu(self.bn_h1(self.conv_h1(group1)))
        l1 = self.relu(self.bn_l1(self.conv_l1(group2)))
        x1 = torch.cat([h1, l1], dim=1)

        # Stage 2 (SeparableConv = depthwise + pointwise)
        h2 = self.relu(self.bn_h2(self.pw_h2(self.dw_h2(x1))))
        l2 = self.relu(self.bn_l2(self.pw_l2(self.dw_l2(x1))))
        x2 = torch.cat([h2, l2], dim=1)

        # Stage 3
        h3 = self.relu(self.bn_h3(self.conv_h3(x2)))
        l3 = self.relu(self.bn_l3(self.conv_l3(x2)))
        out = torch.cat([h3, l3], dim=1)

        return out


# ===========================
# MassAtt: Mixed Attention Module
# ===========================
class MassAtt(nn.Module):
    """
    Mixed Attention: combines Channel Attention (SE-like) and Spatial Attention.
    """
    def __init__(self, in_channels: int, ratio: int = 4):
        super(MassAtt, self).__init__()
        # Channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)

        # Spatial attention
        self.spatial_conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
        self.spatial_conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)
        self.spatial_deconv1 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spatial_deconv2 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Channel attention
        ca = self.global_avg_pool(x).view(B, C)
        ca = self.relu(self.fc1(ca))
        ca = self.sigmoid(self.fc2(ca)).view(B, C, 1, 1)

        # Spatial attention
        sa = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        sa = self.relu(self.spatial_conv1(sa))
        sa = self.relu(self.spatial_conv2(sa))
        sa = self.relu(self.spatial_deconv1(sa))
        sa = self.sigmoid(self.spatial_deconv2(sa))

        # Fuse
        attention = ca * sa
        return attention


# ===========================
# PWFS: Pixel-Wise Feature Selection
# ===========================
class PWFS(nn.Module):
    """
    Pixel-Wise Feature Selection:
    Splits feature map into 3 sub-groups channel-wise,
    computes median and max, then averages them.
    """
    def __init__(self):
        super(PWFS, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into 3 equal channel groups
        split1, split2, split3 = torch.chunk(x, 3, dim=1)

        # Compute median via min/max trick
        min_val = torch.min(torch.min(split1, split2), split3)
        max_val = torch.max(torch.max(split1, split2), split3)
        median_val = split1 + split2 + split3 - min_val - max_val

        # Average of max and median
        out = 0.5 * (max_val + median_val)
        return out


# ===========================
# Main LANMSFF Model
# ===========================
class LANMSFF(nn.Module):
    """
    Lightweight Attention Network with Multi-Scale Feature Fusion for FER.

    Args:
        num_classes: Number of emotion classes (default: 8 for FER).
        input_channels: Input image channels (default: 1 for grayscale, 3 for RGB).
    """
    def __init__(self, num_classes: int = 8, input_channels: int = 1):
        super(LANMSFF, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 66, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(66),
            nn.ReLU(inplace=True),
            # SeparableConv2D equivalent
            nn.Conv2d(66, 66, kernel_size=3, padding=1, groups=66, bias=False),
            nn.Conv2d(66, 66, kernel_size=1, bias=False),
            nn.BatchNorm2d(66),
            nn.ReLU(inplace=True),
            nn.Conv2d(66, 66, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(66),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # Block 2: TwoPathConv + MassAtt
        self.block2_two_path = TwoPathConv(66, 72, kernel_size=3, stride=1, padding=1)
        self.block2_mass_att = MassAtt(72, ratio=4)
        self.block2_conv = nn.Conv2d(72, 72, kernel_size=1, padding=0, bias=False)
        self.block2_bn = nn.BatchNorm2d(72)
        self.block2_pool = nn.MaxPool2d(2)
        self.block2_dropout = nn.Dropout(0.4)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(72, 78, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(78),
            nn.ReLU(inplace=True),
            nn.Conv2d(78, 78, kernel_size=3, padding=1, groups=78, bias=False),
            nn.Conv2d(78, 78, kernel_size=1, bias=False),
            nn.BatchNorm2d(78),
            nn.ReLU(inplace=True),
            nn.Conv2d(78, 78, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(78),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # Block 4: TwoPathConv + MassAtt
        self.block4_two_path = TwoPathConv(78, 84, kernel_size=3, stride=1, padding=1)
        self.block4_mass_att = MassAtt(84, ratio=4)
        self.block4_conv = nn.Conv2d(84, 84, kernel_size=1, padding=0, bias=False)
        self.block4_bn = nn.BatchNorm2d(84)
        self.block4_pool = nn.MaxPool2d(2)
        self.block4_dropout = nn.Dropout(0.4)

        # PWFS modules (reduces channels to 1/3 each)
        self.pwfs1 = PWFS()
        self.pwfs2 = PWFS()
        self.pwfs3 = PWFS()

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Compute fused feature dimension
        # b1: 66 -> PWFS -> 22,  b2: 72 -> PWFS -> 24
        # b3: 78 -> PWFS -> 26,  b4: 84 (no PWFS)
        fused_dim = (66 // 3) + (72 // 3) + (78 // 3) + 84  # = 156

        # Classifier
        self.classifier = nn.Linear(fused_dim, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        b1 = self.block1(x)

        # Block 2
        b2i = self.block2_two_path(b1)
        b2_att = self.block2_mass_att(b2i)
        b2 = b2_att * b2i
        b2 = self.block2_conv(b2)
        b2 = self.block2_bn(b2)
        b2 = self.block2_pool(b2)
        b2 = F.relu(b2, inplace=True)
        b2 = self.block2_dropout(b2)

        # Block 3
        b3 = self.block3(b2)

        # Block 4
        b4i = self.block4_two_path(b3)
        b4_att = self.block4_mass_att(b4i)
        b4 = b4_att * b4i
        b4 = self.block4_conv(b4)
        b4 = self.block4_bn(b4)
        b4 = self.block4_pool(b4)
        b4 = F.relu(b4, inplace=True)
        b4 = self.block4_dropout(b4)

        # PWFS
        b1_pwfs = self.pwfs1(b1)
        b2_pwfs = self.pwfs2(b2)
        b3_pwfs = self.pwfs3(b3)

        # Global pooling + flatten
        b1_gap = self.gap(b1_pwfs).flatten(1)
        b2_gap = self.gap(b2_pwfs).flatten(1)
        b3_gap = self.gap(b3_pwfs).flatten(1)
        b4_gap = self.gap(b4).flatten(1)

        # Multi-scale feature fusion
        fused = torch.cat([b1_gap, b2_gap, b3_gap, b4_gap], dim=1)

        # Classification
        output = self.classifier(fused)
        return output


# ===========================
# Alias for convenience
# ===========================
Net = LANMSFF  # For compatibility with `from external.LANMSFF.model import Net`


if __name__ == "__main__":
    # Quick test
    model = LANMSFF(num_classes=8, input_channels=1)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
