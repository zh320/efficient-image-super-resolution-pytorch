from .sr_base_dataset import SRBaseDataset


class SRDataset(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'train': ['train/T91',
                                  'train/General100',
                                  'train/BSD200',],

                        'val': ['val/BSD100',
                                'val/Set5',
                                'val/Set14',],
                        }

        super(SRDataset, self).__init__(config, data_split, mode)
