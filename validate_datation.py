import os
import wandb
import sys

from aac_metrics import Evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import time
import tqdm

from transformers import BertTokenizer

from models import utils, caption
from datasets import dew
from datasets.utils import read_json, tkn
from configuration import Config

import argparse

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


def validate(config, args):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoints = ['multimodal_datation_test_False_False'] # [f'multimodal_datation_test_{si}_{sc}' for si, sc in
                  #  zip([True, True, False, False], [True, False, True, False])]

    model, _ = caption.build_model(config, multimodal=True)
    dataset_val = dew.build_dataset(config, mode='validation')
    criterion = (utils.ndcg, utils.year_dif)

    for experiment in checkpoints:

        checkpoint_path = f'checkpoints/{experiment}.pth'
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
        mean_ndcg = np.array([0., 0.])
        length = 0
        for j, data in enumerate(data_loader_val):
            vals = evaluate_datation(model, data, criterion, device=config.device)
            mean_ndcg += vals
            length = j
        mean_ndcg /= length
        print(f"NDCG validation mean {mean_ndcg[0]}"
              f"\nYear difference mean {mean_ndcg[1]}")
        with open("metrics_date_estimation.tsv", 'w') as f:
            f.write("ndcg\tdiff\n")
            f.write(f"{mean_ndcg[0]}\t{mean_ndcg[1]}\n")
    return


@torch.no_grad()
def evaluate_datation(model, data, criterion, device="cuda"):
    tasks, images, masks, targets = data

    samples = utils.NestedTensor(images, masks).to(device)

    outputs = model(tasks[0], samples)
    targets = (targets - 1930) // 5
    targets = targets.to(device)

    val = list()
    for crit in criterion:
        val.append(crit(outputs, targets).item())
    return val


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Validate datation model', add_help=False)
    parser.add_argument('--batch_size', default=None, type=int)
    args = parser.parse_args()

    config = Config()
    validate(config, args)
