import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannleSelfAttention(nn.Module):
    def __init__(self, 
                 channels_in : int, 
                 vec_len : int, 
                 num_heads = 8, 
                 is_multi_dir = False):
        super(ChannleSelfAttention, self).__init__()
        self.is_multi_dir = is_multi_dir
        if is_multi_dir:
            vec_len *= 3
        self.pos_embedding = nn.Parameter(torch.randn(1, channels_in, vec_len))
        self.qkv = nn.Linear(vec_len, vec_len*3, bias=False)
        self.num_heads = num_heads
        head_dim = vec_len // num_heads
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(vec_len)

    def forward(self, x):
        origin_shape = x.shape
        if self.is_multi_dir:
            y = x.permute(0,3,2,1)
            z = x.permute(0,2,1,3)
            x = torch.cat([x, y, z], dim=1)
        x = x.view(*(x.shape[0:2]), -1)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        dots = (q @ k.transpose(-2,-1)) * self.scale
        attend = nn.Softmax(dim=-1)(dots)
        x = (attend @ v).transpose(1,2).reshape(B, N, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = x.view(origin_shape)
        x = nn.Sigmoid()(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, out_height, out_width, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            self.CSA = ChannleSelfAttention(out_height * out_width, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        maxtrix_attend = self.CSA(x)
        return x * maxtrix_attend


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
