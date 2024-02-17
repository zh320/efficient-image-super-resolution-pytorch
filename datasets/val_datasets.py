from .sr_base_dataset import SRBaseDataset


class Set5(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                f'val/Set5/image_SRF_{config.upscale}',
                                ],
                        }

        super(Set5, self).__init__(config, data_split, mode)


class Set14(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                f'val/Set14/image_SRF_{config.upscale}',
                                ],
                        }

        super(Set14, self).__init__(config, data_split, mode)


class BSD100(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                f'val/BSD100/image_SRF_{config.upscale}',
                                ],
                        }

        super(BSD100, self).__init__(config, data_split, mode)