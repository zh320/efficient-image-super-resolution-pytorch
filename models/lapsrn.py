"""
Paper:      Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
Url:        https://arxiv.org/abs/1704.03915
Create by:  zh320
Date:       2023/12/16
"""

import torch.nn as nn
from math import log2

from .modules import ConvAct, Upsample


class LapSRN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, hid_channels=64, fe_layer_num=8,
                    act_type='leakyrelu', upsample_type='deconvolution'):
        super(LapSRN, self).__init__()
        assert fe_layer_num > 3, 'Layer number should be larger than 3.\n'
        if upscale in [2, 4, 8]:
            self.num_stage = int(log2(upscale))
            scale_factor = 2
        elif upscale == 3:
            self.num_stage = 1
            scale_factor = 3
        else:
            raise ValueError(f'Unsupported scale factor: {upscale}\n')

        self.fe_branch = FeatureExtraction(in_channels, out_channels, hid_channels, self.num_stage,
                                            fe_layer_num, scale_factor, upsample_type, act_type)
        self.ir_branch = ImageReconstruction(in_channels, out_channels, self.num_stage, 
                                            scale_factor, upsample_type)

    def forward(self, x):
        feats = self.fe_branch(x)
        x = self.ir_branch(x, feats)

        return x


class FeatureExtraction(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch, num_stage, layer_num, scale_factor, 
                    upsample_type, act_type):
        super(FeatureExtraction, self).__init__()
        self.num_stage = num_stage
        self.conv = nn.ModuleList()
        self.out = nn.ModuleList()
        for i in range(num_stage):
            init_ch = in_ch if i==0 else hid_ch
            layers = [ConvAct(init_ch, hid_ch, 3, act_type=act_type)]
            for _ in range(layer_num - 3):
                layers.append(ConvAct(hid_ch, hid_ch, 3, act_type=act_type))
            layers.append(Upsample(hid_ch, hid_ch, scale_factor, upsample_type))

            self.conv.append(nn.Sequential(*layers))

            self.out.append(ConvAct(hid_ch, out_ch, 3, act_type=act_type))

    def forward(self, x):
        feats = []
        for i in range(self.num_stage):
            x = self.conv[i](x)
            feat = self.out[i](x)
            feats.append(feat)

        return feats


class ImageReconstruction(nn.Module):
    def __init__(self, in_ch, out_ch, num_stage, scale_factor, upsample_type):
        super(ImageReconstruction, self).__init__()
        self.num_stage = num_stage
        self.up = nn.ModuleList()
        for i in range(num_stage):
            init_ch = in_ch if i == 0 else out_ch
            self.up.append(Upsample(init_ch, out_ch, scale_factor, upsample_type))

    def forward(self, img, feats):
        for i in range(self.num_stage):
            img = self.up[i](img)
            img += feats[i]

        return img
