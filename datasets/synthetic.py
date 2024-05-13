from torch.utils.data import Dataset
from torch import Tensor
import torch
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import pandas as pd
import random
import os
import tqdm

import random

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
    def __init__(self, angles=(0, 90, 180, 270)):
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


class SyntheticCaption(Dataset):
    def __init__(self, root, data, max_length, limit, transform=train_transform, mode='training', token=False):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(row[0]), row[1]) for row in data.itertuples(index=False)]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]
        self.token = token

        # Initialize tokenizer (adding language and named entity tokens)
        langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
        additional_tokens_dict = {'additional_tokens': additional_tokens}
        special_tokens_dict = {'additional_special_tokens': tkn(langs)}
        self.tokenizer.add_tokens(additional_tokens_dict)
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_length = max_length + 1

    def _process(self, image_id):
        if str(image_id)[-1] == 'g':
            return image_id
        dir_id = int(image_id) // 4096
        return os.path.join(str(dir_id), str(image_id) + '.jpg')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        if self.token:
            caption = f'{tkn(self.token)} {caption}'
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True,
            return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return 'captioning', image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, synthetic_images=True, synthetic_captions=False, mode='training', token=False):
    root = config.dir
    image = 'synthetic_image' if synthetic_images else 'image'
    token = token if not token else ('ca' if synthetic_captions else 'en')

    if mode == 'training':
        train_dir = os.path.join(root, 'historic_sd/images' if synthetic_images else 'coco2017/train')
        train_file = os.path.join(root, 'coco2017', 'captions_train_cat.tsv' if synthetic_captions else
                                  'captions_train.tsv')
        df = pd.read_csv(train_file, sep='\t')[[image, 'caption']]
        df.columns = [['image', 'caption']]
        data = SyntheticCaption(train_dir, df, max_length=config.max_position_embeddings, limit=config.limit,
                                transform=train_transform, mode='training', token=token)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(root, 'historic_sd/images' if synthetic_images else 'coco2017/test')
        val_file = os.path.join(root, 'coco2017', 'captions_test_cat.tsv' if synthetic_captions else
                                'captions_test.tsv')
        df = pd.read_csv(val_file, sep='\t')[[image, 'caption']]
        df.columns = [['image', 'caption']]
        data = SyntheticCaption(val_dir, df, max_length=config.max_position_embeddings, limit=config.limit,
                                transform=val_transform, mode='validation', token=token)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
