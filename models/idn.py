"""
Paper:      Fast and Accurate Single Image Super-Resolution via Information Distillation Network
Url:        https://arxiv.org/abs/1803.09454
Create by:  zh320
Date:       2023/12/30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvAct, Upsample


class IDN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, num_blocks=4, D3=64, s=4,
                    act_type='leakyrelu', upsample_type='deconvolution'):
        super(IDN, self).__init__()
        assert s > 1, 's should be larger than 1, otherwise split_ratio will be out of range.\n'
        split_ratio = 1 / s
        d = int(split_ratio * D3)
        self.upscale = upscale

        self.fblock = nn.Sequential(
                            ConvAct(in_channels, D3, 3, act_type=act_type),
                            ConvAct(D3, D3, 3, act_type=act_type)
                        )

        layers = []
        for i in range(num_blocks):
            layers.append(DBlock(D3, d, act_type))
        self.dblocks = nn.Sequential(*layers)

        self.rblock = Upsample(D3, out_channels, upscale, upsample_type, 17)
    
    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')

        x = self.fblock(x)
        x = self.dblocks(x)
        x = self.rblock(x)

        x += x_up

        return x


class DBlock(nn.Sequential):
    def __init__(self, D3, d, act_type):
        super(DBlock, self).__init__(
            EnhancementUnit(D3, d, act_type),
            conv1x1(D3 + d, D3)
        )


class EnhancementUnit(nn.Module):
    def __init__(self, D3, d, act_type, groups=[1,4,1,4,1,1]):
        super(EnhancementUnit, self).__init__()
        assert len(groups) == 6, 'Length of groups should be 6.\n'
        self.d = d

        self.conv1 = nn.Sequential(
                            ConvAct(D3, D3 - d, 3, groups=groups[0], act_type=act_type),
                            ConvAct(D3 - d, D3 - 2*d, 3, groups=groups[1], act_type=act_type),
                            ConvAct(D3 - 2*d, D3, 3, groups=groups[2], act_type=act_type),
                        )

        self.conv2 = nn.Sequential(
                            ConvAct(D3 - d, D3, 3, groups=groups[3], act_type=act_type),
                            ConvAct(D3, D3 - d, 3, groups=groups[4], act_type=act_type),
                            ConvAct(D3 - d, D3 + d, 3, groups=groups[5], act_type=act_type),
                        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x_c = x[:, :self.d, :, :]
        x_c = torch.cat([x_c, residual], dim=1)
        x_s = x[:, self.d:, :, :]
        x_s = self.conv2(x_s)

        return x_s + x_c
