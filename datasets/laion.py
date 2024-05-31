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


class LAIONCaption(Dataset):
    def __init__(self, root, ann, lang, ner, max_length, transform=train_transform):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = ann
        if type(lang) == list:
            self.lang = '_'.join(lang)
        else:
            self.lang = lang
        self.lang = lang
        self.ner = ner

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

    def _process(self, image_id):
        return str(image_id) + '.jpg'

    def __getitem__(self, idx):
        image_id, caption, lang = self.annot[idx]
        caption = f'{tkn(lang)} {caption}'
        image = Image.open(os.path.join(self.root, self._process(image_id)))

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


def build_dataset(config, lang, ner, mode='training'):
    root = os.path.join(config.dir, 'laion')
    if mode == 'training':
        train_dir = os.path.join(root, 'img_size')
        train_file = os.path.join(root, 'train_coincidences.csv' if not ner else 'train_ner.tsv')
        ann = pd.read_csv(train_file, sep='\t')
        if isinstance(lang, list):
            idx = [i for i, l in enumerate(ann['LANGUAGE']) if l in lang]
            ann = ann.iloc[idx]
        else:
            ann = ann[ann['LANGUAGE'] == lang]
        ann = [(img, caption[:config.max_position_embeddings], la) for img, caption, la in
               zip(ann['SAMPLE_ID'], ann['TEXT'], ann['LANGUAGE'])]
        data = LAIONCaption(train_dir, ann, lang=lang, ner=ner, max_length=config.max_position_embeddings,
                            transform=train_transform)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(root, 'crossmodal_imgs')
        val_file = os.path.join(root, 'captions.tsv')
        ann = pd.read_csv(val_file, sep='\t')
        if isinstance(lang, list):
            idx = [i for i, l in enumerate(ann['lang']) if l in lang]
            ann = ann.iloc[idx]
        else:
            ann = ann[ann['lang'] == lang]
        ann = [(img, caption[:config.max_position_embeddings], la) for img, caption, la in
               zip(ann['image/key'], ann['caption'], ann['lang'])]
        data = LAIONCaption(val_dir, ann, lang=lang, ner=ner, max_length=config.max_position_embeddings,
                            transform=val_transform)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
