from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
from torch import Tensor
import torch

from PIL import Image
import numpy as np
import pandas as pd
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


class XACCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training', token=False):
        super().__init__()

        self.root = root
        self.transform = transform
        data = pd.read_csv(ann, sep='\t')
        self.annot = [(img, caption) for img, caption in zip(data['image'], data['caption'])]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]
        self.token = token

        # Initialize tokenizer (adding language and named entity tokens)
        langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
        special_tokens_dict = {'additional_special_tokens': tkn(langs)}
        self.tokenizer.add_tokens(additional_tokens)
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        if self.token:
            caption = f'<ca> {caption}'
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


def build_dataset(config, ner=False, mode='training', token=False, data=None):
    root = os.path.join(config.dir, 'xac')
    if mode == 'training':
        train_dir = os.path.join(root, 'images')
        train_file = os.path.join(root, 'captions_train.tsv' if not ner else 'captions_train_ner.tsv')
        data = XACCaption(train_dir, train_file, max_length=config.max_position_embeddings, limit=config.limit,
                          transform=train_transform, mode='training', token=token)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(root, 'images')
        val_file = os.path.join(root, data if data is not None else
                                ('captions_test.tsv' if not ner else 'captions_test_ner.tsv'))
        data = XACCaption(val_dir, val_file, max_length=config.max_position_embeddings, limit=config.limit,
                          transform=val_transform, mode='validation', token=token)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
