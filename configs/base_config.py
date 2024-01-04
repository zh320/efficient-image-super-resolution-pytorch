class BaseConfig:
    def __init__(self,):
        # Dataset
        self.dataset = None
        self.dataroot = None
        self.upscale = 2
        self.train_y = True

        # Model
        self.model = None
        self.in_channels = 1
        self.out_channels = 1

        # Training
        self.total_epoch = 3200
        self.base_lr = 0.001
        self.train_bs = 16      # For each GPU
        self.early_stop_epoch = 1000
        self.max_itrs_per_epoch = 200
        
        # Validating
        self.val_bs = 1        # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation
        self.metrics = 'psnr'
        
        # Testing
        self.is_testing = False
        self.test_bs = 1
        self.test_data_folder = None
        
        # Benchmark
        self.benchmark = False
        self.benchmark_datasets = ['set5', 'set14', 'bsd100']

        # Loss
        self.loss_type = 'mse'
        
        # Scheduler
        self.lr_policy = 'constant'
        self.step_size = 2e5        # For step lr scheduler
        self.step_gamma = 0.5       # For step lr scheduler
        self.warmup_epochs = 3      # For cos_warmup lr scheduler
        
        # Optimizer
        self.optimizer_type = 'adam'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD
        
        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        
        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = True
        self.ema_decay = 0.999
        self.ema_start_epoch = 0

        # Augmentation
        self.patch_size = None
        self.rotate = 0.0
        self.multi_scale = False
        self.hflip = 0.0
        self.vflip = 0.0

        # DDP
        self.synBN = False

    def init_dependent_config(self):
        if self.load_ckpt_path is None and not (self.is_testing or self.benchmark):
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if (self.is_testing or self.benchmark) and (self.load_ckpt_path is None):
            self.load_ckpt_path = 'best.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size, self.patch_size]
