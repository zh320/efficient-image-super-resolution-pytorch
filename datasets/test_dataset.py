import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .sr_base_dataset import SRBaseDataset


class TestDataset(Dataset):
    def __init__(self, config):
        data_folder = os.path.expanduser(config.test_data_folder)
        self.train_y = config.train_y
        self.scale = config.upscale
        self.test_lr = config.test_lr

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Test image directory: {data_folder} does not exist.')

        self.hr_images = []
        self.img_names = []

        for file_name in os.listdir(data_folder):
            self.hr_images.append(os.path.join(data_folder, file_name))
            self.img_names.append(file_name)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr = Image.open(self.hr_images[index]).convert('RGB')
        img_name = self.img_names[index]

        if self.test_lr:
            # Resize image to make it compatible for upscale factor if needed
            hr_width = (hr.width // self.scale) * self.scale
            hr_height = (hr.height // self.scale) * self.scale
            if hr_width != hr.width or hr_height != hr.height:
                hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)

            # Generate low resolution image using bicubic interpolation of HR image
            lr_width = hr_width // self.scale
            lr_height = hr_height // self.scale
            lr = hr.resize((lr_width, lr_height), resample=Image.BICUBIC)
            
            # Need interpolated CbCr channels to recover hr images if train with y channel
            bicubic = lr.resize((lr.width * self.scale, lr.height * self.scale), resample=Image.BICUBIC)
        else:   # test hr
            bicubic = hr.resize((hr.width * self.scale, hr.height * self.scale), resample=Image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        bicubic = np.array(bicubic).astype(np.float32)

        if self.test_lr:
            lr = np.array(lr).astype(np.float32)
            if self.train_y:
                # RGB to YCbCr (get interpolated CbCr channels here)
                ycbcr = SRBaseDataset.rgb_to_ycbcr(bicubic)

        if self.train_y:
            # RGB to YCbCr (only need Y channel here)
            hr = SRBaseDataset.rgb_to_ycbcr(hr, y_only=True)
            if self.test_lr:
                lr = SRBaseDataset.rgb_to_ycbcr(lr, y_only=True)

            # HW to CHW --> normalize
            hr = np.expand_dims(hr / 255., 0)
            if self.test_lr:
                lr = np.expand_dims(lr / 255., 0)
        else:
            # HWC to CHW --> normalize
            hr = hr.transpose((2, 0, 1)) / 255.
            if self.test_lr:
                lr = lr.transpose((2, 0, 1)) / 255.

        images = [np.ascontiguousarray(hr), bicubic]
        if self.test_lr:
            images.append(np.ascontiguousarray(lr))

        if self.train_y:
            images.append(np.ascontiguousarray(ycbcr))

        return images, img_name
