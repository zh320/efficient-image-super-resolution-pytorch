"""
Paper:      Image Super-Resolution Using Deep Convolutional Networks
Url:        https://arxiv.org/abs/1501.00092
Create by:  zh320
Date:       2023/12/09
"""

import torch.nn as nn
import torch.nn.functional as F

from .modules import conv5x5, ConvAct


class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, kernel_setting='935', 
                    act_type='relu'):
        super(SRCNN, self).__init__()
        if kernel_setting not in ['915', '935', '955']:
            raise ValueError(f'Unknown kernel setting: {kernel_setting}. You can choose \
                                from ["915", "935", "955"].\n')
        kernel_map = {'915':1, '935':3, '955':5}

        self.upscale = upscale
        self.layer1 = ConvAct(in_channels, 64, 9, act_type=act_type)
        self.layer2 = ConvAct(64, 32, kernel_map[kernel_setting], act_type=act_type)
        self.layer3 = conv5x5(32, out_channels)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
