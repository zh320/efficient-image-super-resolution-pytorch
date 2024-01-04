import os, torch
from .carn import CARN
from .drcn import DRCN
from .drrn import DRRN
from .edsr import EDSR
from .espcn import ESPCN
from .fsrcnn import FSRCNN
from .idn import IDN
from .lapsrn import LapSRN
from .srcnn import SRCNN
from .vdsr import VDSR


model_hub = {'carn':CARN, 'drcn':DRCN, 'drrn':DRRN, 'edsr':EDSR, 'espcn':ESPCN, 'fsrcnn':FSRCNN, 
             'idn':IDN, 'lapsrn':LapSRN, 'srcnn':SRCNN, 'vdsr':VDSR,}


def get_model(config):
    if config.model in model_hub.keys():
        model = model_hub[config.model](in_channels=config.in_channels, 
                                        out_channels=config.out_channels, 
                                        upscale=config.upscale)
    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model
