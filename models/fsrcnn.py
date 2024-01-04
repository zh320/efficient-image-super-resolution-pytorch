"""
Paper:      Accelerating the Super-Resolution Convolutional Neural Network
Url:        https://arxiv.org/abs/1608.00367
Create by:  zh320
Date:       2023/12/09
"""

import torch.nn as nn

from .modules import ConvAct, Upsample


class FSRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, d=56, s=12, act_type='prelu', 
                    upsample_type='deconvolution'):
        super(FSRCNN, self).__init__()
        self.first_part = ConvAct(in_channels, d, 5, act_type=act_type, num_parameters=d)
        self.mid_part = nn.Sequential(
                            ConvAct(d, s, 1, act_type=act_type, num_parameters=s),
                            ConvAct(s, s, 3, act_type=act_type, num_parameters=s),
                            ConvAct(s, s, 3, act_type=act_type, num_parameters=s),
                            ConvAct(s, s, 3, act_type=act_type, num_parameters=s),
                            ConvAct(s, s, 3, act_type=act_type, num_parameters=s),
                            ConvAct(s, d, 1, act_type=act_type, num_parameters=d)
                        )
        self.last_part = Upsample(d, out_channels, upscale, upsample_type, 9)
    
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
