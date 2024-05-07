import torch
import torch.nn as nn
from typing import Optional, List
from torch import Tensor
from torch.utils.data import DataLoader

from .synthetic import build_dataset as bd_synthetic
from .xac import build_dataset as bd_xac
from .dew import build_dataset as bd_dew
from .laion import build_dataset as bd_laion

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


def build_dataloader(dataset, datation, config, lang=None, ner=False, synthetic_images=False, synthetic_captions=False,
                     weighted_criterion=False):
    if dataset == 'synthetic':
        dataset_train = bd_synthetic(
            config, synthetic_images=synthetic_images,
            synthetic_captions=synthetic_captions, mode='training')
        dataset_val = bd_synthetic(
            config, synthetic_images=synthetic_images,
            synthetic_captions=synthetic_captions, mode='validation')
    elif dataset == 'xac':
        dataset_train = bd_xac(config, ner=ner, mode='training')
        dataset_val = bd_xac(config, ner=ner, mode='validation')
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

    if weighted_criterion:
        weights = dataset_train.get_weights(config.new_vocab_size, synthetic_captions).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        return data_loader_train, data_loader_val, criterion

    return data_loader_train, data_loader_val, None
