import torch 
from torch import nn
from lib.lightrfb import LightRFB

class AttentionDownBlock(nn.Module):
    def __init__(self, channels_in, channels_out, frame_channels):
        super(AttentionDownBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, frame_channels, 1, bias=False),
        )
        channels_mid = (channels_in + channels_out) // 2
        self.rfb = LightRFB(channels_in, channels_mid, channels_out)
    
    def forward(self, x):
        x_frame = self.mlp(x)
        x_down = self.rfb(x)

        return x_frame, x_down

    
class FusionConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(FusionConv, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.fusion(x)


class FrameAttention(nn.Module):
    def __init__(self, channels_in, frame_channels, ratio):
        super(FrameAttention, self).__init__()
        self.block1 = AttentionDownBlock(channels_in, channels_in // ratio, frame_channels)
        channels_in = channels_in // ratio
        self.block2 = AttentionDownBlock(channels_in, channels_in // ratio, frame_channels)
        channels_in = channels_in // ratio
        self.block3 = AttentionDownBlock(channels_in, frame_channels, frame_channels)
        self.block4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(frame_channels, frame_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(frame_channels // ratio, frame_channels, 1, bias=False),
        )
        self.fusion = FusionConv(frame_channels * 4, frame_channels, 1)

    def forward(self, x, origin_shape):
        frame_feature1 = x.view((origin_shape[0]), -1, *(x.shape[2:]))
        frame_weight1, frame_feature2 = self.block1(frame_feature1)
        frame_weight2, frame_feature3 = self.block2(frame_feature2)
        frame_weight3, frame_feature4 = self.block3(frame_feature3)
        frame_weight4 = self.block4(frame_feature4)
        frame_weight_fusion = torch.cat([frame_weight1, frame_weight2, frame_weight3, frame_weight4], dim=1)
        frame_weight_fusion = self.fusion(frame_weight_fusion)

        return frame_weight_fusion.view(-1, 1, 1, 1)



