import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time
import sys

from models import utils, caption
from datasets import coco, synthetic, xac
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

        checkpoint_path = f'checkpoints/{args.checkpoint}.pth'
        model, criterion = caption.build_model(config)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            raise NotImplementedError('Incorrect checkpoint from coco, hist_sd or xac')
        model.load_state_dict(checkpoint['model'])
        model.to(device)

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
            dataset_train = synthetic.build_dataset(
                config, synthetic_images=True, synthetic_captions=False, mode='training')
            dataset_val = synthetic.build_dataset(
                config, synthetic_images=True, synthetic_captions=False, mode='validation')
        elif args.dataset == 'xac':
            dataset_train = xac.build_dataset(config, mode='training')
            dataset_val = xac.build_dataset(config, mode='validation')

        print(f"Train: {len(dataset_train)}")
        print(f"Valid: {len(dataset_val)}")

        if args.weighted_criterion:
            weights = dataset_train.get_weights(config.new_vocab_size).to(config.device)
            criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, config.batch_size, drop_last=True
        )

        data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
        data_loader_val = DataLoader(
            dataset_val, config.batch_size, sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

        print("Start Training..")
        global_loss = dict()
        global_validation_loss = dict()
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            epoch_loss = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
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
            }, os.path.join('checkpoints', f'checkpoint_from_{args.checkpoint}_to_{args.dataset}.pth'))

            validation_loss = evaluate(model, criterion, data_loader_val, device)
            for task in validation_loss.keys():
                print(f"Validation Loss for {task}: {validation_loss[task]}")
                try:
                    global_validation_loss[task].append(validation_loss[task])
                except:
                    global_validation_loss[task] = [validation_loss[task]]
            wandb.log({f'epoch_validation_loss_{task}': validation_loss[task] for task in validation_loss.keys()})

            print()

        with open(f'checkpoints/epoch_loss_from_{args.checkpoint}_to_{args.dataset}.tsv', 'w') as f:
            f.write('epoch\tloss\n')
            f.write('\n'.join([f'{i + 1}\t{loss}' for i, loss in enumerate(global_loss)]))


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser('Finetune model', add_help=False)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default='coco', type=str)
    parser.add_argument('--weighted_criterion', default=False, type=bool)
    args = parser.parse_args()

    config = Config()
    finetune(config, args)
