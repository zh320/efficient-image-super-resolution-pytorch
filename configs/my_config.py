from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'sr'
        self.data_root = '/path/to/your/dataset'
        self.upscale = 2
        self.train_y = True

        # Model
        self.model = 'srcnn'
        self.in_channels = 1
        self.out_channels = 1

        # Training
        self.total_epoch = 6400
        self.lr_policy = 'constant'         # or step
        self.logger_name = 'sr_trainer'

        # Augmentation
        self.patch_size = 48
        self.rotate = 0.5
        self.multi_scale = True
