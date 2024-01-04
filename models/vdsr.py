"""
Paper:      Accurate Image Super-Resolution Using Very Deep Convolutional Networks
Url:        https://arxiv.org/abs/1511.04587
Create by:  zh320
Date:       2023/12/16
"""

import torch.nn as nn
import torch.nn.functional as F

from .modules import conv3x3, ConvAct


class VDSR(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, layer_num=20, hid_channels=64, 
                    act_type='relu'):
        super(VDSR, self).__init__()
        self.upscale = upscale
        self.first_layer = conv3x3(in_channels, hid_channels)
        layers = [ConvAct(hid_channels, hid_channels, 3, inplace=True) for i in range(layer_num-2)]
        self.mid_layer = nn.Sequential(*layers)
        self.last_layer = conv3x3(hid_channels, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
        res = self.first_layer(x)
        res = self.mid_layer(res)    
        res = self.last_layer(res)
        res += x

        return res
