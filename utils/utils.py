import os, random, torch, json
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer
    
    
def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    if config.benchmark:
        keys = ['benchmark_datasets', 'upscale', 'train_y', 'model', 'in_channels', 'out_channels',]
    else:
        keys = ['dataset', 'upscale', 'train_y', 'model', 'in_channels', 'out_channels', 'loss_type', 
                'optimizer_type', 'lr_policy', 'total_epoch', 'lr', 'train_bs', 'val_bs',  
                'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
                'DDP', 'use_ema', 'patch_size', 'multi_scale']

    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)
