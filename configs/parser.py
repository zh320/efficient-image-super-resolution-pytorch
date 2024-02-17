import argparse


def load_parser(config):
    args = get_parser()

    for k,v in vars(args).items():
        if v is not None:
            try:
                exec(f"config.{k} = v")
            except:
                raise RuntimeError(f'Unable to assign value to config.{k}')
    return config


def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default=None, choices=['cityscapes'],
        help='choose which dataset you want to use')
    parser.add_argument('--dataroot', type=str, default=None, 
        help='path to your dataset')
    parser.add_argument('--upscale', type=int, default=None, 
        help='scale factor for super resolution')
    parser.add_argument('--train_y', action='store_false', default=None, 
        help='whether to train Y channel in YCbCr space or not (default: True)')

    # Model
    parser.add_argument('--model', type=str, default=None, 
        choices=['carn', 'drcn', 'drrn', 'edsr', 'espcn', 'fsrcnn', 
                 'idn', 'lapsrn', 'srcnn', 'srdensenet', 'vdsr'],
        help='choose which model you want to use')
    parser.add_argument('--in_channels', type=int, default=None, 
        help='number of input channel for given model (default: 1 for Y input else 3 for RGB input)')
    parser.add_argument('--out_channels', type=int, default=None, 
        help='number of output channel for given model (default: 1 for Y input else 3 for RGB input)')

    # Training
    parser.add_argument('--total_epoch', type=int, default=None, 
        help='number of total training epochs')
    parser.add_argument('--base_lr', type=float, default=None, 
        help='base learning rate for single GPU, total learning rate *= gpu number')
    parser.add_argument('--train_bs', type=int, default=None, 
        help='training batch size for single GPU, total batch size *= gpu number')
    parser.add_argument('--early_stop_epoch', type=int, default=None,
        help='epoch number to stop training if validation score does not increase')
    parser.add_argument('--max_itrs_per_epoch', type=int, default=None, 
        help='increase the number of training iterations (suppose there are few training samples)')

    # Validating
    parser.add_argument('--val_bs', type=int, default=None, 
        help='validating batch size for single GPU, total batch size *= gpu number')    
    parser.add_argument('--begin_val_epoch', type=int, default=None, 
        help='which epoch to start validating')    
    parser.add_argument('--val_interval', type=int, default=None, 
        help='epoch interval between two validations')
    parser.add_argument('--metrics', type=str, default=None, choices = ['psnr', 'ssim'],
        help='choose which validation metric you want to use (default: psnr)')

    # Testing
    parser.add_argument('--is_testing', action='store_true', default=None,
        help='whether to perform testing/predicting or not (default: False)')
    parser.add_argument('--test_bs', type=int, default=None, 
        help='testing batch size (currently only support single GPU)')
    parser.add_argument('--test_data_folder', type=str, default=None, 
        help='path to your testing image folder')
    parser.add_argument('--test_lr', action='store_false', default=None,
        help='whether to test the downscaled/low-resolution image or not (default: True)')

    # Benchmark
    parser.add_argument('--benchmark', action='store_true', default=None,
        help='whether to perform benchmarking for given datasets or not (default: False)')
    parser.add_argument('--benchmark_datasets', type=list, default=None, 
        help='select which datasets to benchmark (default: [set5, set14, bsd100])')

    # Loss
    parser.add_argument('--loss_type', type=str, default=None, choices = ['mse', 'mae', 'charbonnier'],
        help='choose which loss you want to use')

    # Scheduler
    parser.add_argument('--lr_policy', type=str, default=None, 
        choices = ['constant', 'step', 'linear', 'cos_warmup'],
        help='choose which learning rate policy you want to use (default: constant)')
    parser.add_argument('--warmup_epochs', type=int, default=None,
        help='warmup epoch number for `cos_warmup` learning rate policy')
    parser.add_argument('--step_size', type=int, default=None, 
        help='step size for `step` learning rate policy')
    parser.add_argument('--step_gamma', type=float, default=None, 
        help='lr reduction factor for `step` learning rate policy (default: 0.1)')

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default=None, 
        choices = ['sgd', 'adam', 'adamw'],
        help='choose which optimizer you want to use (default: adam)')
    parser.add_argument('--momentum', type=float, default=None, 
        help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=None, 
        help='weight decay rate of SGD optimizer')

    # Monitoring
    parser.add_argument('--save_ckpt', action='store_false', default=None,
        help='whether to save checkpoint or not (default: True)')
    parser.add_argument('--save_dir', type=str, default=None, 
        help='path to save checkpoints and training configurations etc.')
    parser.add_argument('--use_tb', action='store_false', default=None,
        help='whether to use tensorboard or not (default: True)')
    parser.add_argument('--tb_log_dir', type=str, default=None, 
        help='path to save tensorboard logs')
    parser.add_argument('--ckpt_name', type=str, default=None, 
        help='given name of the saved checkpoint, otherwise use `last` and `best`')

    # Training setting
    parser.add_argument('--amp_training', action='store_true', default=None,
        help='whether to use automatic mixed precision training or not (default: False)')
    parser.add_argument('--resume_training', action='store_false', default=None,
        help='whether to load training state from specific checkpoint or not if present (default: True)')
    parser.add_argument('--load_ckpt', action='store_false', default=None,
        help='whether to load given checkpoint or not if exist (default: True)')
    parser.add_argument('--load_ckpt_path', type=str, default=None, 
        help='path to load specific checkpoint, otherwise try to load `last.pth`')
    parser.add_argument('--base_workers', type=int, default=None, 
        help='number of workers for single GPU, total workers *= number of GPU')
    parser.add_argument('--random_seed', type=int, default=None, 
        help='random seed')
    parser.add_argument('--use_ema', action='store_false', default=None,
        help='whether to use exponetial moving average to update weights or not (default: True)')
    parser.add_argument('--ema_decay', type=float, default=None, 
        help='constant decay factor for EMA update, if not given use linear decay instead')
    parser.add_argument('--ema_start_epoch', type=int, default=None, 
        help='epoch number to start EMA update')

    # Augmentation
    parser.add_argument('--patch_size', type=int, default=None, 
        help='crop size for single training patch')
    parser.add_argument('--rotate', type=float, default=None, 
        help='probability to perform rotation')
    parser.add_argument('--multi_scale', action='store_false', default=None, 
        help='whether to perform multi-scale training or not (default: False)')
    parser.add_argument('--hflip', type=float, default=None, 
        help='probability to perform horizontal flip')
    parser.add_argument('--vflip', type=float, default=None, 
        help='probability to perform vertical flip')

    # DDP
    parser.add_argument('--synBN', action='store_true', default=None, 
        help='whether to use SyncBatchNorm or not if trained with DDP (default: False)')
    parser.add_argument('--local_rank', type=int, default=None, 
        help='used for DDP, DO NOT CHANGE')

    args = parser.parse_args()
    return args
