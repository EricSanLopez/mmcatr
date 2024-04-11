import numpy as np
import pandas as pd
import wandb
import time
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import utils, caption
from datasets.dataloader import build_dataloader
from configuration import Config
from engine import train_one_epoch, evaluate

import argparse
import warnings

warnings.filterwarnings('ignore')


def main(config, args):
    with wandb.init(project="XAC-ImageCaptioning",
                    name=f'{config.experiment}_{args.synthetic_images}_{args.synthetic_captions}',
                    config=config):

        config = wandb.config  # access all HPs through wandb.config, so logging matches execution
        device = torch.device(config.device)
        print(f'Initializing Device: {device}')

        seed = config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, criterion = caption.build_model(config, args.date_estimation)
        model.to(device)

        criterion_datation = utils.get_criterion("Triplet")

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

        data_loader_train, data_loader_val = build_dataloader(
            args.dataset, args.date_estimation, config, args.synthetic_images, args.synthetic_captions)

        if os.path.exists(config.checkpoint):
            print("Loading Checkpoint...")
            checkpoint = torch.load(config.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.start_epoch = checkpoint['epoch'] + 1

        print("Start Training..")
        global_loss = dict()
        global_validation_loss = dict()
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            epoch_loss = train_one_epoch(
                model, criterion, criterion_datation, data_loader_train,
                optimizer, device, epoch, config.clip_max_norm)
            lr_scheduler.step()
            for task in epoch_loss.keys():
                print(f"Training Loss for {task}: {epoch_loss[task]}")
                try:
                    global_loss[task].append(epoch_loss[task])
                except:
                    global_loss[task] = [epoch_loss[task]]
            wandb.log({f'epoch_loss_{task}': epoch_loss[task] for task in epoch_loss.keys()})

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join('checkpoints',
                            f'{config.experiment}_{args.synthetic_images}_{args.synthetic_captions}.pth'))

            validation_loss = evaluate(model, criterion, data_loader_val, device)
            for task in validation_loss.keys():
                print(f"Validation Loss for {task}: {validation_loss[task]}")
                try:
                    global_validation_loss[task].append(validation_loss[task])
                except:
                    global_validation_loss[task] = [validation_loss[task]]
            wandb.log({f'epoch_validation_loss_{task}': validation_loss[task] for task in validation_loss.keys()})

            print()

        with open(f'checkpoints/{config.experiment}_{args.synthetic_images}_{args.synthetic_captions}.tsv', 'w') as f:
            f.write('epoch\ttraining\tvalidation\n')
            f.write('\n'.join([f'{i + 1}\t{losses[0]}\t{losses[1]}' for i, losses in
                               enumerate(zip(global_loss, global_validation_loss))]))


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser('Train model from zero', add_help=False)
    parser.add_argument('-si', '--synthetic_images', default=False, type=bool)
    parser.add_argument('-sc', '--synthetic_captions', default=False, type=bool)
    parser.add_argument('-d', '--dataset', default='synthetic', type=str)
    parser.add_argument('-e', '--date_estimation', default=False, type=bool)
    args = parser.parse_args()

    config = Config()
    main(config, args)
