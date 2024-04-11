import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import math
import sys
import tqdm
import argparse
import numpy as np
import pandas as pd
from operator import itemgetter

from transformers import BertTokenizer
from models import utils, caption
from datasets import synthetic, xac
from configuration import Config
from datasets.utils import read_json, tkn

import warnings

warnings.filterwarnings('ignore')


def main(config, args):
    with wandb.init(project="XAC-ImageCaptioning", name=config.experiment, config=config):  # Starting wandb
        config = wandb.config  # access all HPs through wandb.config, so logging matches execution
        device = torch.device(config.device)
        print(f'Initializing Device: {device}')

        seed = config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, criterion = caption.build_model(config)
        model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
        model.transformer.embeddings.word_embeddings = nn.Embedding(
            config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        model.to(device)

        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(f"Number of params: {n_parameters}")

        param_dicts = [
            {"params": [p for n, p in model.named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": config.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=config.lr, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

        dataset_train = synthetic.build_dataset(
            config, synthetic_images=True, synthetic_captions=False, mode='training')
        print(f"Train: {len(dataset_train)}")

        sampler_train = torch.utils.data.SequentialSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, config.batch_size, drop_last=True
        )

        data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)

        if os.path.exists(args.checkpoint):
            print("Loading Checkpoint...")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # config.start_epoch = checkpoint['epoch'] + 1

        langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        special_tokens = ['<loc>', '<per>', '<org>', '<misc>']
        special_tokens_dict = {'additional_special_tokens': special_tokens + tkn(langs)}
        tokenizer.add_special_tokens(special_tokens_dict)

        weights = evaluate(model, criterion, data_loader_train, device, tokenizer)
        df = pd.DataFrame(weights, columns=['loss'])
        df['image_id'] = range(len(weights))
        df[['image_id', 'loss']].to_csv('caption_loss.tsv', sep='\t', index=False)
        # weights = {tokenizer.decode([k]): v for k, v in weights.items()}
        # df = pd.DataFrame(weights.items(), columns=['token', 'mean_loss'])
        # df.to_csv('losses_per_cap.tsv', sep='\t', index=False)
        return


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, tokenizer):
    model.eval()
    criterion.eval()

    # weights = {v: [0, 0] for v in range(config.new_vocab_size)}
    weights = list()

    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])

            for sample, cap in zip(outputs, caps):
                loss = criterion(sample, cap[1:])
                # idx = torch.argmax(sample, dim=1).tolist()
                # weights.update(zip(idx, [[a+loss.item(), b+1] for a, b in itemgetter(*idx)(weights)]))
                # weights.update(zip(cap.tolist(), [[a+loss.item(), b+1] for a, b in itemgetter(*cap.tolist())(weights)]))
                weights.append(loss.item())

            pbar.update(1)

    # weights = {k: v[0] / v[1] if v[1] > 0 else 0 for k, v in weights.items()}
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
    args = parser.parse_args()
    wandb.login()

    config = Config()
    main(config, args)
