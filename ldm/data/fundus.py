import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class fundus(Dataset):
    def __init__(self, data_root, image_size=512, image_number=1000):
        self.blur_paths = sorted(make_dataset(os.path.join(data_root, "0.25"), image_number))
        self.clear_paths = sorted(make_dataset(os.path.join(data_root, "raw"), image_number))
        self._length = len(self.blur_paths)
        self.image_size = image_size
        # self.transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        blur_path = self.blur_paths[i]
        blur = Image.open(blur_path).convert('RGB')
        blur = blur.resize((self.image_size, self.image_size), Image.BICUBIC)
        blur = np.array(blur).astype(np.uint8)
        blur = (blur / 127.5 - 1.0).astype(np.float32)

        clear_path = self.clear_paths[i]
        clear = Image.open(clear_path).convert('RGB')
        clear = clear.resize((self.image_size, self.image_size), Image.BICUBIC)
        clear = np.array(clear).astype(np.uint8)
        clear = (clear / 127.5 - 1.0).astype(np.float32)

        return {'blur': blur, 'clear': clear, "clear_path": clear_path, "blur_path": blur_path}

class DRIVE(Dataset):
    def __init__(self, data_root_blur, data_root_clear, image_size=512, image_number=1000):
        self.blur_paths = sorted(make_dataset(data_root_blur, image_number))
        self.clear_paths = sorted(make_dataset(data_root_clear, image_number))
        self._length = len(self.blur_paths)
        self.image_size = image_size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        blur_path = self.blur_paths[i]
        blur = Image.open(blur_path).convert('RGB')
        blur = blur.resize((self.image_size, self.image_size), Image.BICUBIC)
        blur = np.array(blur).astype(np.uint8)
        blur = (blur / 127.5 - 1.0).astype(np.float32)

        clear_path = self.clear_paths[i]
        clear = Image.open(clear_path).convert('RGB')
        clear = clear.resize((self.image_size, self.image_size), Image.BICUBIC)
        clear = np.array(clear).astype(np.uint8)
        clear = (clear / 127.5 - 1.0).astype(np.float32)

        return {'blur': blur, 'clear': clear, "clear_path": clear_path, "blur_path": blur_path}


class fundusReconstruction(Dataset):
    def __init__(self, data_root, image_size=512, image_number=1000):
        self.image_paths = sorted(make_dataset(data_root, image_number))
        self._length = len(self.image_paths)
        self.image_size = image_size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return {'image': image}
    

class fundusTrain(fundus):
    def __init__(self, **kwargs):
        super().__init__(data_root = "/home/lhq323/Projects/degradation_1017/datasets/HQI/HQI7Level/HQI7Level/train", **kwargs)

class fundusValidation(fundus):
    def __init__(self, **kwargs):
        super().__init__(data_root = "/home/lhq323/Projects/degradation_1017/datasets/HQI/HQI7Level/HQI7Level/test", **kwargs)
