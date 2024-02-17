from .sr_base_dataset import SRBaseDataset


class SRDataset(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'train': ['train/BSDS200',
                                  'train/General100',
                                  'train/T91',],

                        'val': [f'val/BSD100/image_SRF_{config.upscale}',
                                f'val/Set5/image_SRF_{config.upscale}',
                                f'val/Set14/image_SRF_{config.upscale}',],
                        }

        super(SRDataset, self).__init__(config, data_split, mode)
