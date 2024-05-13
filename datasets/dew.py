import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, read_json, tkn

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float64)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DewDatation(Dataset):
    def __init__(self, root, data, transform=train_transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def _process(self, image_id):
        return os.path.join(str(image_id)[0], str(image_id)[1:3], str(image_id))

    def __getitem__(self, idx):
        target, image_id = self.data.iloc[idx]
        image = Image.open(os.path.join(self.root, self._process(image_id) + '.jpg'))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        return 'datation', image.tensors.squeeze(0), image.mask.squeeze(0), target

    def get_image(self, idx):
        target, image_id = self.data.iloc[idx]
        image = Image.open(os.path.join(self.root, self._process(image_id) + '.jpg'))
        return image, target


def build_dataset(config, mode='training'):
    root = os.path.join(config.dir, 'dew')
    if mode == 'training':
        train_dir = os.path.join(root, 'images')
        train_file = os.path.join(root, 'gt_train_ok.csv')
        data = DewDatation(train_dir, pd.read_csv(train_file, names=['target', 'file'])[:591752],
                           transform=train_transform)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(root, 'images')
        val_file = os.path.join(root, 'gt_test_ok.csv')
        data = DewDatation(val_dir, pd.read_csv(val_file, names=['target', 'file']), transform=val_transform)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
