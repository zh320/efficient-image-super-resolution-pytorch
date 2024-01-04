from .sr_base_dataset import SRBaseDataset


class Set5(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                'val/Set5',
                                ],
                        }

        super(Set5, self).__init__(config, data_split, mode)


class Set14(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                'val/Set14',
                                ],
                        }

        super(Set14, self).__init__(config, data_split, mode)


class BSD100(SRBaseDataset):
    def __init__(self, config, mode):
        data_split =  {
                        'val': [
                                'val/BSD100',
                                ],
                        }

        super(BSD100, self).__init__(config, data_split, mode)