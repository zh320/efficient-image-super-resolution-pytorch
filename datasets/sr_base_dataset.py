import os, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SRBaseDataset(Dataset):
    IMG_SUFFIX = ['jpg', 'jpeg', 'png', 'bmp']

    def __init__(self, config, data_split, mode='train'):
        assert mode in ['train', 'val'], f'Unsupported dataset mode: {mode}.\n'

        data_root = os.path.expanduser(config.data_root)
        img_dirs = []
        for img_dir in data_split[mode]:
            img_dirs.append(os.path.join(data_root, img_dir))

        if len(img_dirs) == 0:
            raise RuntimeError('No image directory found.')

        for img_dir in img_dirs:
            if not os.path.isdir(img_dir):
                raise RuntimeError(f'Image directory: {img_dir} does not exist.')

        self.mode = mode
        self.train_y = config.train_y
        self.scale = config.upscale
        self.patch_size = config.patch_size
        self.random_rotate = config.rotate
        self.multi_scale = config.multi_scale
        self.hflip = config.hflip
        self.vflip = config.vflip

        hr_images = []
        for img_dir in img_dirs:
            for file_name in os.listdir(img_dir):
                if file_name.split('.')[-1].lower() in SRBaseDataset.IMG_SUFFIX:
                    img_path = os.path.join(img_dir, file_name)
                    hr_images.append(img_path)

        if self.mode == 'train':
            dataset_repeat_times = config.max_itrs_per_epoch*config.train_bs*config.gpu_num // len(hr_images) 
            hr_images *= dataset_repeat_times        

        self.hr_images = hr_images

        self.num = len(self.hr_images)

    def rgb_to_y(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError(f'\nInput must be np.ndarray, but got type: {type(img)} instead.')
        if img.shape[2] != 3:
            raise ValueError(f'\nInput should be RGB channel array, but got {img.shape[2]} channel instead.')

        img = np.dot(img, [65.481, 128.553, 24.966]) / 255. + 16.
        return img.astype(np.float32)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr = Image.open(self.hr_images[index]).convert('RGB')
        
        if self.mode == 'train':
            # Perform multiscale augmentation
            if self.multi_scale:
                # In case the image is too small to perform random crop
                min_scale = max(self.patch_size[0]/hr.width, self.patch_size[1]/hr.height)
                if min_scale > 1:
                    scale = min_scale
                else:
                    scale = random.uniform(0.75, 1.0)

                new_w = int(hr.width * scale)
                new_h = int(hr.height * scale)
                hr = hr.resize((new_w, new_h), resample=Image.BICUBIC)

            # Calculate the random crop location x0
            if hr.width > self.patch_size[0]:
                x0 = random.randint(0, hr.width-self.patch_size[0])
            else:
                x0 = 0

            # Calculate the random crop location y0
            if hr.height > self.patch_size[1]:
                y0 = random.randint(0, hr.height-self.patch_size[1])
            else:
                y0 = 0

            # Random crop a path using patch_size
            hr = hr.crop([x0, y0, x0+self.patch_size[0], y0+self.patch_size[1]])

            # Random rotation
            if random.random() < self.random_rotate:
                angle = random.randint(0, 3) * 90
                hr = hr.rotate(angle, expand=True)

            # Random horizontal flip
            if random.random() < self.hflip:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip
            if random.random() < self.vflip:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)

        # Resize image to make it compatible for training if needed
        hr_width = (hr.width // self.scale) * self.scale
        hr_height = (hr.height // self.scale) * self.scale
        if hr_width != hr.width or hr_height != hr.height:
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)

        # Generate low resolution image using bicubic interpolation of HR image
        lr_width = hr_width // self.scale
        lr_height = hr_height // self.scale
        lr = hr.resize((lr_width, lr_height), resample=Image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        if self.train_y:
            # RGB to YCbCr (only need Y channel here)
            hr = self.rgb_to_y(hr)
            lr = self.rgb_to_y(lr)

            # HW to CHW --> normalize
            hr = np.expand_dims(hr / 255., 0)
            lr = np.expand_dims(lr / 255., 0)
        else:
            # HWC to CHW --> normalize
            hr = hr.transpose((2, 0, 1)) / 255.
            lr = lr.transpose((2, 0, 1)) / 255.

        return np.ascontiguousarray(lr), np.ascontiguousarray(hr)
