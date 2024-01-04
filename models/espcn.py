"""
Paper:      Real-Time Single Image and Video Super-Resolution Using an Efficient 
            Sub-Pixel Convolutional Neural Network
Url:        https://arxiv.org/abs/1609.05158
Create by:  zh320
Date:       2023/12/09
"""

import torch.nn as nn

from .modules import ConvAct, Activation, Upsample


class ESPCN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, n1=64, n2=32, act_type='tanh', 
                    upsample_type='pixelshuffle'):
        super(ESPCN, self).__init__()
        self.layer1 = ConvAct(in_channels, n1, 5, act_type=act_type)
        self.layer2 = ConvAct(n1, n2, 3, act_type=act_type)
        self.upsample = Upsample(n2, out_channels, upscale, upsample_type, 3)
        self.act3 = Activation(act_type)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.act3(self.upsample(x))
        return x
