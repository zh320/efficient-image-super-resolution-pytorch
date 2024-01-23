import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=0.01):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred, label):
        loss = torch.sqrt((pred - label)**2 + self.eps).mean()
        return loss.mean()


def get_loss_fn(config, device):
    if config.loss_type == 'mae':
        criterion = nn.L1Loss()
    
    elif config.loss_type == 'mse':
        criterion = nn.MSELoss()
        
    elif config.loss_type == 'charbonnier':
        criterion = CharbonnierLoss()

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")
        
    return criterion
