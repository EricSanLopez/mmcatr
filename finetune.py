import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time
import sys

from models import utils, caption
from datasets import coco, hist_sd, xac
from configuration import Config
from engine import train_one_epoch, evaluate

import argparse

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


def finetune(config, args):
    with wandb.init(project="XAC-ImageCaptioning", name=f'Finetune_from_{args.checkpoint}_to_{args.dataset}',
                    config=config):  # Starting wandb
        device = torch.device(config.device)
        print(f'Initializing Device: {device}')

        seed = config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, criterion = caption.build_model(config)
        if args.checkpoint == 'coco':
            checkpoint_path = 'checkpoints/checkpoint_coco_from_zero.pth'
            model, _ = caption.build_model(config)
            model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
            model.transformer.embeddings.word_embeddings = nn.Embedding(
                config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.to(device)
        elif args.checkpoint == 'hist_sd':
            checkpoint_path = 'checkpoints/checkpoint_hist_sd_from_zero.pth'
            model, _ = caption.build_model(config)
            model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
            model.transformer.embeddings.word_embeddings = nn.Embedding(
                config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.to(device)
        elif args.checkpoint == 'xac':
            checkpoint_path = 'checkpoints/checkpoint_xac_from_zero.pth'
            model, _ = caption.build_model(config)
            model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
            model.transformer.embeddings.word_embeddings = nn.Embedding(
                config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.to(device)
        else:
            raise NotImplementedError('Incorrect checkpoint from coco, hist_sd or xac')

        config.lr = 1e-5
        config.epochs = args.epochs
        config.lr_drop = 8

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

        if args.dataset == 'coco':
            dataset_train = coco.build_dataset(config, mode='training')
            dataset_val = coco.build_dataset(config, mode='validation')
        elif args.dataset == 'hist_sd':
            dataset_train = hist_sd.build_dataset(config, mode='training')
            dataset_val = hist_sd.build_dataset(config, mode='validation')
        elif args.dataset == 'xac':
            dataset_train = xac.build_dataset(config, mode='training')
            dataset_val = xac.build_dataset(config, mode='validation')
        print(f"Train: {len(dataset_train)}")
        print(f"Valid: {len(dataset_val)}")

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, config.batch_size, drop_last=True
        )

        data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
        data_loader_val = DataLoader(dataset_val, config.batch_size,
                                     sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

        """
        if os.path.exists(config.checkpoint):
            print("Loading Checkpoint...")
            checkpoint = torch.load(config.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.start_epoch = checkpoint['epoch'] + 1
        """

        print("Start Training..")
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            epoch_loss = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
            lr_scheduler.step()
            print(f"Training Loss: {epoch_loss}")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join('checkpoints', f'checkpoint_from_{args.checkpoint}_to_{args.dataset}.pth'))

            validation_loss = evaluate(model, criterion, data_loader_val, device)
            print(f"Validation Loss: {validation_loss}")

            print()


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser('Finetune model', add_help=False)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default='coco', type=str)
    args = parser.parse_args()

    config = Config()
    finetune(config, args)
