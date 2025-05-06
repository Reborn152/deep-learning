import cv2
import torch
import torch.nn as nn
from einops import rearrange


class SwinIR(nn.Module):
    """轻量级SwinIR超分辨率模型（优化版）"""

    def __init__(self, upscale=4, in_chans=3, embed_dim=48):
        super().__init__()
        self.upscale = upscale

        # 浅层特征提取
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 残差Swin块
        self.layers = nn.Sequential(
            *[ResidualSwinBlock(embed_dim, window_size=8) for _ in range(4)]
        )

        # 上采样层
        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.layers(x)
        return self.upsampler(x)


class ResidualSwinBlock(nn.Module):
    """优化的残差Swin块"""

    def __init__(self, dim, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads=4)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 窗口注意力
        x = self.norm1(x)
        x = self.attn(x) + x

        # MLP
        x = self.norm2(x)
        x = self.mlp(x) + x

        return shortcut + x.permute(0, 3, 1, 2)


class WindowAttention(nn.Module):
    """高效窗口注意力机制"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # 划分窗口
        x = rearrange(x, 'b (h ws1) (w ws2) c -> b h w (ws1 ws2) c',
                      ws1=self.ws, ws2=self.ws)

        # 多头注意力
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, H // self.ws, W // self.ws, self.ws * self.ws, C)

        # 重组窗口
        x = rearrange(x, 'b h w (ws1 ws2) c -> b (h ws1) (w ws2) c', ws1=self.ws)
        return self.proj(x)


class ViT_FaceNet(nn.Module):
    """轻量级人脸特征提取器"""

    def __init__(self, feat_dim=512):
        super().__init__()
        # MobileNetV3作为骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # 112x112
            nn.Hardswish(),
            nn.Conv2d(16, 32, 3, 2, 1),  # 56x56
            nn.Hardswish(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 28x28
            nn.AdaptiveAvgPool2d(1)  # 全局池化
        )
        self.head = nn.Linear(64, feat_dim)

    @staticmethod
    def preprocess(img):
        """标准化预处理"""
        img = cv2.resize(img, (112, 112))  # 固定输入尺寸
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (img_tensor - 0.5) / 0.5  # 归一化到[-1, 1]

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x.view(x.size(0), -1))