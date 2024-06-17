import os
import wandb
import sys

from aac_metrics import Evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision
from torchvision import models

import numpy as np
import pandas as pd
import time
import tqdm
import pickle
from PIL import Image

from transformers import BertTokenizer

from models import utils, caption
from datasets import dew
from datasets.utils import read_json, tkn
from configuration import Config
from annoy_index import annoy

import argparse

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torchvision.models.resnet101(pretrained=True)
        self.conv.fc = nn.Linear(2048, 128)
        self.reg = nn.Linear(128, 1, bias=True)

    def forward(self, x):
        y = self.conv(x)
        y_value = self.reg(y) + 1975
        return y


def validate(config, args):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoints = [f'multimodal_experiment_False_{sc}' for sc in [False, True]] if args.checkpoint is None \
        else [args.checkpoint]

    dataset_val = dew.build_dataset(config, mode='validation')

    if args.classifier:
        validate_classifier(checkpoints, dataset_val, args, config)
        return

    model, _ = caption.build_model(config, multimodal=True)
    criterion = (utils.ndcg, utils.year_dif)

    for experiment in checkpoints:

        ann = generate_annoy(config, experiment)
        checkpoint_path = f'/data2fast/users/esanchez/checkpoints/{experiment}.pth'  # .pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            raise NotImplementedError('Incorrect checkpoint path')
        model.load_state_dict(checkpoint['model'])

        model.to(device)
        model.eval()

        print(f"Valid: {len(dataset_val)}")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(dataset_val, args.batch_size if args.batch_size is not None else config.batch_size,
                                     sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

        print(f"Start validation..")
        mean_ndcg = np.array([0., 0., 0.])
        length = 0
        for j, data in enumerate(data_loader_val):
            vals, difs = evaluate_datation(model, data, criterion, ann, device=config.device)
            mean_ndcg += [*vals, difs]
            length = j
        mean_ndcg /= length
        print(f"NDCG validation mean {mean_ndcg[0]}\n"
              f"Year difference mean {mean_ndcg[1]}\n"
              f"Annoy year difference mean {mean_ndcg[2]}\n")
        with open(f"logs/metrics_date_estimation_{experiment}.tsv", 'w') as f:
            f.write("ndcg\tdiff\tannoy\n")
            f.write(f"{mean_ndcg[0]}\t{mean_ndcg[1]}\t{mean_ndcg[2]}\n")
    return


@torch.no_grad()
def evaluate_datation(model, data, criterion, ann, device="cuda"):
    tasks, images, masks, targets = data
    samples = utils.NestedTensor(images, masks).to(device)
    outputs = model(tasks[0], samples)
    # outputs = model(images.to(device))

    dif = list()
    for i, (output, target) in enumerate(zip(outputs, targets)):
        # nns = (ann.retrieve_by_vector(output.detach().cpu().numpy()) - 1930) // 5
        nns = ann.retrieve_by_vector(output.detach().cpu().numpy())
        dif.append(np.abs(nns - target.detach().cpu().numpy()))

    dif = np.mean(dif)

    val = list()
    for crit in criterion:
        val.append(crit(outputs, targets).item())

    return val, dif


def generate_annoy(config, checkpoint_path='multimodal_experiment_False_False'):
    model, criterion = caption.build_model(config, multimodal=True)
    try:
        checkpoint = torch.load(f"/data2fast/users/esanchez/checkpoints/{checkpoint_path}.pth", map_location='cpu')
    except:
        raise NotImplementedError('Incorrect checkpoint from coco, hist_sd or xac')
    model.load_state_dict(checkpoint['model'])
    model.to('cuda')
    model.eval()

    dataset_train = dew.build_dataset(config, mode="training")
    sampler_train_dew = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train_dew = DataLoader(
        dataset_train, 256, sampler=sampler_train_dew, num_workers=config.num_workers, drop_last=False)

    ann = annoy.Annoyer(
        model, data_loader_train_dew, emb_size=256, out_dir=os.path.join("annoy_index", checkpoint_path)
    )

    try:
        ann.load()
    except OSError:
        ann.state_variables['built'] = False
        ann.fit()
    return ann


def validate_classifier(checkpoints, dataset, args, config):

    model = models.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 70)

    for experiment in checkpoints:

        checkpoint_path = f'{experiment}.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            raise NotImplementedError('Incorrect checkpoint path')
        model.load_state_dict(checkpoint['model'])

        model.to(config.device)
        model.eval()

        print(f"Valid: {len(dataset)}")

        sampler_val = torch.utils.data.SequentialSampler(dataset)

        data_loader_val = DataLoader(dataset, args.batch_size if args.batch_size is not None else config.batch_size,
                                     sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

        print(f"Start validation..")
        mean_ndcg = np.array([0., 0.])
        length = 0
        for j, data in enumerate(data_loader_val):
            dif, dif_5 = evaluate_classifier(model, data, device=config.device)
            mean_ndcg += [dif, dif_5]
            length = j
        mean_ndcg /= length
        print(f"Year difference mean {mean_ndcg[0]}\n"
              f"Year difference mean (grouped by 5 years) {mean_ndcg[1]}\n")
        with open(f"logs/metrics_date_estimation_{experiment}.tsv", 'w') as f:
            f.write("diff\tdiff_5\n")
            f.write(f"{mean_ndcg[0]}\t{mean_ndcg[1]}\n")
    return


@torch.no_grad()
def evaluate_classifier(model, data, device="cuda"):
    tasks, images, masks, targets = data
    images = images.to(device)
    outputs = model(images)

    targets = targets.detach().cpu().numpy() - 1930
    outputs = torch.argmax(outputs, dim=1)
    outputs = outputs.detach().cpu().numpy()

    dif = np.abs(outputs - targets)

    targets_5 = targets // 5
    outputs_5 = outputs // 5

    dif_5 = np.abs(outputs_5 - targets_5)

    return np.mean(dif), np.mean(dif_5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Validate datation model', add_help=False)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--classifier', default=False, type=bool)
    args = parser.parse_args()

    config = Config()
    validate(config, args)
