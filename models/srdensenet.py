"""
Paper:      Image Super-Resolution Using Dense Skip Connections
Url:        https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
Create by:  zh320
Date:       2024/01/27
"""

import torch
import torch.nn as nn

from .modules import conv3x3, ConvAct, Activation, Upsample


class SRDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, hid_channels=128, num_block=8, num_layer=8, 
                    act_type='relu', upsample_type='deconvolution'):
        super(SRDenseNet, self).__init__()
        assert upscale in [2,3,4], f'Unsupported upscale factor: {upscale}.\n'
        self.num_block = num_block

        # Initial Convolution
        self.conv = ConvAct(in_channels, hid_channels, 3, act_type=act_type)

        # Dense Blocks
        self.dense_blocks = nn.ModuleList([])
        for _ in range(num_block):
            self.dense_blocks.append(DenseBlock(hid_channels, hid_channels, num_layer, act_type))

        # Bottleneck Layer
        self.bottleneck = ConvAct(hid_channels*(num_block+1), hid_channels*2, 1, act_type=act_type)

        # Deconvolution Layers
        if upscale in [2, 3]:
            self.deconvolution = Upsample(hid_channels*2, hid_channels*2, upscale, upsample_type)
        elif upscale in [4]:
            self.deconvolution = nn.Sequential(
                                 Upsample(hid_channels*2, hid_channels*2, 2, upsample_type),
                                 Upsample(hid_channels*2, hid_channels*2, 2, upsample_type)
                                )

        # Reconstruction Layer
        self.reconstruction = conv3x3(hid_channels*2, out_channels)

    def forward(self, x):
        x = self.conv(x)

        feats = [x]
        for i in range(self.num_block):
            x = self.dense_blocks[i](x)
            feats.append(x)

        x = self.bottleneck(torch.cat(feats, dim=1))
        x = self.deconvolution(x)
        x = self.reconstruction(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, act_type):
        super(DenseBlock, self).__init__()
        assert out_channels % num_layer == 0, 'out_channels should be evenly divided by num_layer.\n'
        self.num_layer = num_layer
        growth_rate = out_channels // num_layer

        self.conv0 = conv3x3(in_channels, growth_rate)
        self.act = nn.ModuleList([Activation(act_type) for _ in range(num_layer)])
        self.conv = nn.ModuleList([])
        for i in range(1, num_layer-1):
            self.conv.append(conv3x3(i*growth_rate, growth_rate))
        self.conv.append(conv3x3((num_layer-1)*growth_rate, out_channels))

    def forward(self, x):
        x = self.conv0(x)
        feats = [x]

        for i in range(self.num_layer - 1):
            x = torch.cat(feats, dim=1)
            x = self.act[i](x)
            feat = self.conv[i](x)
            if i != self.num_layer - 1:
                feats.append(feat)

        feat = self.act[-1](feat)

        return feat
