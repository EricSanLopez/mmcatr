import torch
import torch.nn as nn
from typing import Optional, List
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

from .synthetic import build_dataset as bd_synthetic
from .xac import build_dataset as bd_xac
from .dew import build_dataset as bd_dew
from .laion import build_dataset as bd_laion
from .utils import read_json, tkn

import json
import os


class MultitaskLoader:
    def __init__(self, data_loaders, batch_size):
        self.data_loaders = data_loaders
        self.iterators = [iter(data_loader) for data_loader in self.data_loaders]
        self.sizes = [len(data_loader.dataset) for data_loader in data_loaders]
        self.idx = [0] * len(self.sizes)
        self.max_idx = [size // batch_size for size in self.sizes]
        self.probs = self.sizes.copy()

    def __iter__(self):
        return self

    def __next__(self):
        if not any(self.probs):
            self.reset_index()
            raise StopIteration
        task_idx = torch.multinomial(torch.tensor([float(s) for s in self.probs]), 1).item()
        self.idx[task_idx] += 1
        if self.idx[task_idx] == self.max_idx[task_idx]:
            self.probs[task_idx] = 0
        return next(self.iterators[task_idx])

    def __len__(self):
        return sum(self.max_idx)

    def reset_index(self):
        self.probs = self.sizes.copy()
        self.idx = [0] * len(self.sizes)
        self.iterators = [iter(data_loader) for data_loader in self.data_loaders]


def get_tokenizer():
    # Initialize tokenizer (adding language and named entity tokens)
    langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
    special_tokens_dict = {'additional_special_tokens': tkn(langs)}
    tokenizer.add_tokens(additional_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_weights(dataset, data_loaders, size, lang=None, ner=None):
    dataset.sort()
    name = f'weights'
    for ds in dataset:
        if ds == 'laion':
            if isinstance(lang, list):
                lang = '_'.join(lang)
            name += f'_{ds}_{lang}' + ('_ner' if ner else '')
        elif ds == 'synthetic':
            name += f'_{ds}' + (f'_{lang}' if lang is not None else '')
        else:  # xac
            name += f'_{ds}' + ('_ner' if ner else '')
    name += '.pth'

    try:
        weights = torch.load(os.path.join('/data2fast/users/esanchez/checkpoints', name))['weights']
    except FileNotFoundError:
        tokenizer = get_tokenizer()
        if isinstance(data_loaders, list):
            data = data_loaders[0].dataset.annot
            for dl in data_loaders[1:]:
                data.extend(dl.dataset.annot)
        else:
            data = data_loaders.dataset.annot
        weights = [0] * size
        for ann in data:
            caption = ann[1]
            tokens = tokenizer.encode(caption)
            for i in tokens:
                weights[i] += 1
        weights = Tensor(1 - np.array(weights) / sum(weights))

        # NER tokens at max
        # weights[119547: 119547 + 4] = 1

        torch.save({
            'weights': weights
        }, os.path.join('/data2fast/users/esanchez/checkpoints', name))

    return weights


def build_dataloader(dataset, datation, config, lang=None, ner=False, synthetic_images=False,
                     weighted_criterion=False, token=False, data=None):
    coco, xac = False, False
    if isinstance(lang, str) and len(lang) > 2:
        token = True
        lang = lang.split(',')
        if 'en' in lang:
            lang.remove('en')
            coco = True
        if 'ca' in lang:
            lang.remove('ca')
            xac = True

    if dataset == 'synthetic':
        dataset_train = bd_synthetic(
            config, synthetic_images=synthetic_images, lang=lang, mode='training', token=token)
        dataset_val = bd_synthetic(
            config, synthetic_images=synthetic_images, lang=lang, mode='validation', token=token)
    elif dataset == 'xac':
        dataset_train = bd_xac(config, ner=ner, mode='training', token=token, data=data)
        dataset_val = bd_xac(config, ner=ner, mode='validation', token=token, data=data)
    elif dataset == 'laion':
        dataset_train = bd_laion(config, lang=lang, ner=ner, mode='training')
        dataset_val = bd_laion(config, lang=lang, ner=ner, mode='validation')
    else:
        raise NotImplementedError(f'Dataset {dataset} not in ["synthetic", "xac", "laion"]')

    print(f"Train {dataset}: {len(dataset_train)}")
    print(f"Valid {dataset}: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    if coco:
        dataset_train_coco = bd_synthetic(
            config, synthetic_images=False, mode="training", token=token)
        dataset_val_coco = bd_synthetic(
            config, synthetic_images=False, mode="validation", token=token)

        print(f"Train COCO: {len(dataset_train_coco)}")
        print(f"Valid COCO: {len(dataset_val_coco)}")

        sampler_train_coco = torch.utils.data.RandomSampler(dataset_train_coco)
        sampler_val_coco = torch.utils.data.SequentialSampler(dataset_val_coco)

        batch_sampler_train_coco = torch.utils.data.BatchSampler(
            sampler_train_coco, config.batch_size, drop_last=True
        )

        data_loader_train_coco = DataLoader(
            dataset_train_coco, batch_sampler=batch_sampler_train_coco, num_workers=config.num_workers)
        data_loader_val_coco = DataLoader(
            dataset_val_coco, config.batch_size, sampler=sampler_val_coco, drop_last=False,
            num_workers=config.num_workers)

        data_loader_train = MultitaskLoader([data_loader_train, data_loader_train_coco], config.batch_size)
        data_loader_val = MultitaskLoader([data_loader_val, data_loader_val_coco], config.batch_size)

    if xac:
        dataset_train_xac = bd_xac(config, mode="training", token=token)
        dataset_val_xac = bd_xac(config, mode="validation", token=token)

        print(f"Train XAC: {len(dataset_train_xac)}")
        print(f"Valid XAC: {len(dataset_val_xac)}")

        sampler_train_xac = torch.utils.data.RandomSampler(dataset_train_xac)
        sampler_val_xac = torch.utils.data.SequentialSampler(dataset_val_xac)

        batch_sampler_train_xac = torch.utils.data.BatchSampler(
            sampler_train_xac, config.batch_size, drop_last=True
        )

        data_loader_train_xac = DataLoader(
            dataset_train_xac, batch_sampler=batch_sampler_train_xac, num_workers=config.num_workers)
        data_loader_val_xac = DataLoader(
            dataset_val_xac, config.batch_size, sampler=sampler_val_xac, drop_last=False,
            num_workers=config.num_workers)

        data_loader_train = MultitaskLoader([data_loader_train, data_loader_train_xac], config.batch_size)
        data_loader_val = MultitaskLoader([data_loader_val, data_loader_val_xac], config.batch_size)

    if weighted_criterion:
        dataset = [dataset, 'synthetic'] if coco else ([dataset, 'xac'] if xac else [dataset])
        data = data_loader_train.data_loaders if isinstance(data_loader_train, MultitaskLoader) else data_loader_train
        weights = get_weights(dataset, data, config.new_vocab_size, lang, ner).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    else:
        criterion = None

    if datation:
        dataset_train_dew = bd_dew(config, mode="training")
        dataset_val_dew = bd_dew(config, mode="validation")

        print(f"Train DEW: {len(dataset_train_dew)}")
        print(f"Valid DEW: {len(dataset_val_dew)}")

        sampler_train_dew = torch.utils.data.RandomSampler(dataset_train_dew)
        sampler_val_dew = torch.utils.data.SequentialSampler(dataset_val_dew)

        batch_sampler_train_dew = torch.utils.data.BatchSampler(
            sampler_train_dew, config.batch_size, drop_last=True
        )

        data_loader_train_dew = DataLoader(
            dataset_train_dew, batch_sampler=batch_sampler_train_dew, num_workers=config.num_workers)
        data_loader_val_dew = DataLoader(
            dataset_val_dew, config.batch_size, sampler=sampler_val_dew, drop_last=False,
            num_workers=config.num_workers)

        data_loader_train = MultitaskLoader([data_loader_train, data_loader_train_dew], config.batch_size)
        data_loader_val = MultitaskLoader([data_loader_val, data_loader_val_dew], config.batch_size)

    return data_loader_train, data_loader_val, criterion
