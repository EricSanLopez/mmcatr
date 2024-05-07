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
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class XACCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        data = pd.read_csv(ann, sep='\t')
        self.annot = [(img, caption) for img, caption in zip(data['image'], data['caption'])]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        # Initialize tokenizer (adding language and named entity tokens)
        langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
        additional_tokens_dict = {'additional_tokens': additional_tokens}
        special_tokens_dict = {'additional_special_tokens': tkn(langs)}
        self.tokenizer.add_tokens(additional_tokens_dict)
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def get_weights(self, size, ner):
        try:
            weights = torch.load(os.path.join('checkpoints', f'weights_xac_ner_{ner}.pth'))['weights']
        except FileNotFoundError:
            weights = [0]*size
            for ann in self.annot:
                caption = ann[1]
                tokens = self.tokenizer.encode(caption)
                for i in tokens:
                    weights[i] += 1
            weights = Tensor(1 - np.array(weights) / sum(weights))
            weights[119547: 119547 + 4] = 1

            torch.save({
                'weights': weights
            }, os.path.join('checkpoints', f'weights_xac_ner_{ner}.pth'))

        return weights

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
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


def build_dataset(config, ner=False, mode='training'):
    root = os.path.join(config.dir, 'xac')
    if mode == 'training':
        train_dir = os.path.join(root, 'train')
        train_file = os.path.join(root, 'captions_train_raw.tsv' if not ner else 'captions_train.tsv')
        data = XACCaption(train_dir, train_file, max_length=config.max_position_embeddings, limit=config.limit,
                          transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(root, 'test')
        val_file = os.path.join(root, 'captions_test_raw.tsv' if not ner else 'captions_test.tsv')
        data = XACCaption(val_dir, val_file, max_length=config.max_position_embeddings, limit=config.limit,
                          transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
