
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from lib.lightrfb import LightRFB
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.funet_model import *
from lib.seg_branch import *
from lib.frame_attention import FrameAttention

class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output

class combine_feature(nn.Module):
    def __init__(self):
        super(combine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(32, 16)
        self.up2_low = nn.Conv2d(24, 16, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(16)
        self.up2_act = nn.PReLU(16)
        self.refine = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.PReLU())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        return refine_feature

class FUnet(nn.Module):
    def __init__(self):
        super(FUnet, self).__init__()
        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        self.High_RFB = LightRFB()
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=24)
        self.frame_seg = FrameAttention(120, 5, 2)
        self.seg_branch = UNet(32, 32)
        self.decoder = combine_feature()
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))

    def forward(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)

        low_feature = self.feature_extractor.layer2(x)
        high_feature = self.feature_extractor.layer3(low_feature)
        high_feature = self.High_RFB(high_feature)
        low_feature = self.Low_RFB(low_feature)

        # Unet + CSA + Frame-att
        # frame branch
        frame_weight_fusion = self.frame_seg(low_feature, origin_shape)

        # seg branch
        high_feature = self.seg_branch(high_feature)
        high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                    mode="bilinear",
                                    align_corners=False)
        x = self.decoder(low_feature, high_feature)
        x = self.SegNIN(x)
        x *= frame_weight_fusion
        x = torch.sigmoid(
            F.interpolate(x, size=(origin_shape[3], origin_shape[4]), mode="bilinear", 
            align_corners=False))
        return x

    def load_backbone(self, pretrained_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
