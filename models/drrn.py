"""
Paper:      Image Super-Resolution via Deep Recursive Residual Network
Url:        https://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html
Create by:  zh320
Date:       2023/12/23
"""

import torch.nn as nn
import torch.nn.functional as F

from .modules import conv3x3, ConvAct, ConvBNAct


class DRRN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, B=1, U=9, hid_channels=128, 
                    act_type='relu', use_bn=False):
        super(DRRN, self).__init__()
        self.upscale = upscale
        self.B = B
        self.U = U

        ConvBlock = ConvBNAct if use_bn else ConvAct
        self.init_blocks = nn.ModuleList()
        self.recursive_blocks = nn.ModuleList()
        for i in range(B):
            in_ch = in_channels if i == B-1 else hid_channels
            self.init_blocks.append(ConvBlock(in_ch, hid_channels, 3, act_type=act_type))
            self.recursive_blocks.append(ResidualUnit(hid_channels, act_type, use_bn))
        self.last_layer = conv3x3(hid_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')

        for i in range(self.B):
            res = self.init_blocks[i](x)
            residual = res
            for _ in range(self.U):
                res = self.recursive_blocks[i](res)
                res += residual

        res = self.last_layer(res)
        x = x + res

        return x


class ResidualUnit(nn.Module):
    def __init__(self, channels, act_type, use_bn):
        super(ResidualUnit, self).__init__()
        ConvBlock = ConvBNAct if use_bn else ConvAct
        self.conv1 = ConvBlock(channels, channels, 3, act_type=act_type)
        self.conv2 = ConvBlock(channels, channels, 3, act_type=act_type)

    def forward(self, x):
        res = self.conv2(self.conv1(x))
        res += x

        return res
