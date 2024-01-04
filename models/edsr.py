"""
Paper:      Enhanced Deep Residual Networks for Single Image Super-Resolution
Url:        https://arxiv.org/abs/1707.02921
Create by:  zh320
Date:       2023/12/16
"""

import torch.nn as nn

from .modules import conv3x3, ConvAct, Upsample


class EDSR(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, B=16, F=64, scale_factor=None, 
                    act_type='relu', upsample_type='pixelshuffle'):
        super(EDSR, self).__init__()
        if scale_factor is None:
            scale_factor = 0.1 if B > 16 else 1.0

        self.first_layer = conv3x3(in_channels, F)

        layers = []
        for _ in range(B):
            layers.append(ResidualBlock(F, scale_factor, act_type))
        self.res_layers = nn.Sequential(*layers)
        
        self.mid_layer = conv3x3(F, F)
        self.last_layers = nn.Sequential(
                                Upsample(F, F, upscale, upsample_type, 3),
                                conv3x3(F, out_channels)
                            )

    def forward(self, x):
        x = self.first_layer(x)
        residual = x
        x = self.res_layers(x)
        x = self.mid_layer(x)
        x += residual
        x = self.last_layers(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale_factor, act_type):
        super(ResidualBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
                        ConvAct(channels, channels, 3, act_type=act_type),
                        conv3x3(channels, channels)
                    )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.scale_factor < 1:
            x = x * self.scale_factor
        x += residual

        return x
