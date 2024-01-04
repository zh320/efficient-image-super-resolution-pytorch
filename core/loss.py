import torch.nn as nn


def get_loss_fn(config, device):
    if config.loss_type == 'mae':
        criterion = nn.L1Loss()
    
    elif config.loss_type == 'mse':
        criterion = nn.MSELoss()

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")
        
    return criterion
