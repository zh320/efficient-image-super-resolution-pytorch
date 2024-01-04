"""
Paper:      Deeply-Recursive Convolutional Network for Image Super-Resolution
Url:        https://arxiv.org/abs/1511.04491
Create by:  zh320
Date:       2023/12/23
"""

import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvAct


class DRCN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, recursions=16, 
                    hid_channels=256, act_type='relu', arch_type='advanced'):
        super(DRCN, self).__init__()
        if arch_type not in ['basic', 'advanced']:
            raise ValueError(f'Unsupported model type: {arch_type}\n')
        self.upscale = upscale
        self.recursions = recursions
        self.arch_type = arch_type

        self.embedding_net = nn.Sequential(
                                ConvAct(in_channels, hid_channels, 3, act_type=act_type),
                                ConvAct(hid_channels, hid_channels, 3, act_type=act_type)
                            )
        self.inference_net = ConvAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.reconstruction_net = nn.Sequential(
                                        ConvAct(hid_channels, hid_channels, 3, act_type=act_type),
                                        ConvAct(hid_channels, out_channels, 3, act_type=act_type)
                                    )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
        if self.arch_type == 'advanced':
            skip = x

        x = self.embedding_net(x)

        for i in range(self.recursions):
            x = self.inference_net(x)

            if self.arch_type == 'advanced':
                if i == 0:
                    res = self.reconstruction_net(x + skip)
                else:
                    res += self.reconstruction_net(x + skip)

        if self.arch_type == 'basic':
            res = self.reconstruction_net(x)

        return res
