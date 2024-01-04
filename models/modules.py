import torch.nn as nn


# Regular convolution with kernel size 3x3
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, 
                    padding=2, bias=True)


# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                    padding=1, bias=True)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, upsample_type=None, 
                    kernel_size=None,):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor + 1
            padding = (kernel_size - 1) // 2
            output_padding = scale_factor - 1
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                stride=scale_factor, padding=padding, 
                                                output_padding=output_padding, bias=True)
        elif upsample_type == 'pixelshuffle':
            ks = kernel_size if kernel_size is not None else 3
            padding = (ks - 1) // 2
            self.up_conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels * (scale_factor**2), ks, 1, padding),
                                nn.PixelShuffle(scale_factor)
                            )
        else:
            ks = kernel_size if kernel_size is not None else 3
            padding = (ks - 1) // 2
            self.up_conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, ks, 1, padding),
                                nn.Upsample(scale_factor=scale_factor, mode='bicubic')
                            )

    def forward(self, x):
        return self.up_conv(x)


# Regular convolution -> activation
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, 
                    groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, 
                    groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )        


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }
                        
        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')
        
        self.activation = activation_hub[act_type](**kwargs)
        
    def forward(self, x):
        return self.activation(x)
