"""
Paper:      Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
Url:        https://arxiv.org/abs/1803.08664
Create by:  zh320
Date:       2023/12/30
"""

import torch
import torch.nn as nn

from .modules import conv1x1, conv3x3, ConvAct, Activation, Upsample


class CARN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, arch_type='carn', 
                    hid_channels=64, act_type='relu', upsample_type='pixelshuffle'):
        super(CARN, self).__init__()
        if arch_type not in ['carn', 'carn-m']:
            raise ValueError(f'Unsupported arch_type: {arch_type}\n')
        block = ResidualBlock if arch_type == 'carn' else ResidualEBlock
        
        self.conv1 = conv3x3(in_channels, hid_channels)
        self.cascading_block1 = CascadingBlock(block, hid_channels, act_type)
        self.conv2 = conv1x1(2*hid_channels, hid_channels)
        self.cascading_block2 = CascadingBlock(block, hid_channels, act_type)
        self.conv3 = conv1x1(3*hid_channels, hid_channels)
        self.cascading_block3 = CascadingBlock(block, hid_channels, act_type)
        self.conv4 = conv1x1(4*hid_channels, hid_channels)
        if upscale in [2, 3]:
            self.upsample = nn.Sequential(
                                conv3x3(hid_channels, hid_channels),
                                Upsample(hid_channels, hid_channels, upscale, upsample_type, 3)
                            )
        elif upscale == 4:
            self.upsample = nn.Sequential(
                                conv3x3(hid_channels, hid_channels),
                                Upsample(hid_channels, hid_channels, 2, upsample_type, 3),
                                conv3x3(hid_channels, hid_channels),
                                Upsample(hid_channels, hid_channels, 2, upsample_type, 3)
                            )
        else:
            raise NotImplementedError(f'Unsupported upscale factor: {upscale}\n')
        self.conv_last = conv3x3(hid_channels, out_channels)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x_cb1 = self.cascading_block1(x1)
        x = torch.cat([x1, x_cb1], dim=1)

        x = self.conv2(x)
        x_cb2 = self.cascading_block2(x)
        x = torch.cat([x1, x_cb1, x_cb2], dim=1)

        x = self.conv3(x)
        x_cb3 = self.cascading_block3(x)
        x = torch.cat([x1, x_cb1, x_cb2, x_cb3], dim=1)

        x = self.conv4(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        return x


class CascadingBlock(nn.Module):
    def __init__(self, block, channels, act_type):
        super(CascadingBlock, self).__init__()
        self.res1 = block(channels, act_type)
        self.conv1 = conv1x1(2*channels, channels)
        self.res2 = block(channels, act_type)
        self.conv2 = conv1x1(3*channels, channels)
        self.res3 = block(channels, act_type)
        self.conv3 = conv1x1(4*channels, channels)

    def forward(self, x):
        x0 = x

        x1 = self.res1(x)
        x = torch.cat([x0, x1], dim=1)
        x = self.conv1(x)

        x2 = self.res2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv2(x)

        x = self.res3(x)
        x = torch.cat([x, x0, x1, x2], dim=1)
        x = self.conv3(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, act_type):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
                        ConvAct(channels, channels, 3, act_type=act_type),
                        conv3x3(channels, channels)
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual

        return self.act(x)


class ResidualEBlock(nn.Module):
    def __init__(self, channels, act_type, groups=4):
        super(ResidualEBlock, self).__init__()
        self.conv = nn.Sequential(
                        ConvAct(channels, channels, 3, groups=groups, act_type=act_type),
                        ConvAct(channels, channels, 3, groups=groups, act_type=act_type),
                        conv1x1(channels, channels)
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual

        return self.act(x)
