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
from engine import train_one_epoch, train_one_epoch_multitask, evaluate, evaluate_multitask

import argparse
import warnings
warnings.filterwarnings('ignore')


def main(config, args):
    output_name = get_output_name(args)
    with wandb.init(project="XAC-ImageCaptioning",
                    name=output_name,
                    config=config):

        config = wandb.config  # access all HPs through wandb.config, so logging matches execution
        device = torch.device(config.device)
        print(f'Initializing Device: {device}')

        seed = config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, criterion = caption.build_model(config, multimodal=args.date_estimation)
        model.to(device)

        criterion_datation = utils.get_criterion("Triplet")
        optimizer, lr_scheduler = utils.get_optimizer(model, config)

        data_loader_train, data_loader_val, aux = build_dataloader(
            args.dataset, args.date_estimation, config, args.language, args.ner, args.synthetic_images,
            args.synthetic_captions, args.weighted_criterion, args.token)

        criterion = aux if args.weighted_criterion else criterion

        if os.path.exists(config.checkpoint):
            print("Loading Checkpoint...")
            checkpoint = torch.load(config.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.start_epoch = checkpoint['epoch'] + 1

        print("Start Training..")
        global_loss, global_validation_loss = utils.ArrayStructure(multitask=args.date_estimation), \
            utils.ArrayStructure(multitask=args.date_estimation)

        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")

            # Training
            if args.date_estimation:
                epoch_loss = train_one_epoch_multitask(
                    model, criterion, criterion_datation, data_loader_train, optimizer, device, epoch,
                    config.clip_max_norm)
                lr_scheduler.step()

                global_loss += epoch_loss
                [print(f"Training Loss for {task}: {epoch_loss[task]}") for task in epoch_loss.keys()]
                wandb.log({f'epoch_loss_{task}': epoch_loss[task] for task in epoch_loss.keys()})

            else:
                epoch_loss = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
                lr_scheduler.step()

                global_loss += epoch_loss
                print(f"Training Loss for captioning: {epoch_loss}")
                wandb.log({f'epoch_loss_captioning': epoch_loss})

            # Save model and metadata
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join('checkpoints', output_name + '.pth'))

            # Validation
            if args.date_estimation:
                validation_loss = evaluate_multitask(model, criterion, criterion_datation, data_loader_val, device)

                global_validation_loss += validation_loss
                [print(f"Validation Loss for {task}: {validation_loss[task]}") for task in validation_loss.keys()]
                wandb.log({f'epoch_validation_loss_{task}': validation_loss[task] for task in validation_loss.keys()})

            else:
                validation_loss = evaluate(model, criterion, data_loader_val, device)
                global_validation_loss += validation_loss
                print(f"Validation Loss for captioning: {validation_loss}")
                wandb.log({f'epoch_validation_loss_captioning': validation_loss})

            print()

        utils.save_losses(output_name, global_loss, global_validation_loss)


def get_output_name(args):
    output_name = 'multimodal_' if args.date_estimation else ''
    if args.dataset == 'synthetic':
        output_name += f'{config.experiment}_{args.synthetic_images}_{args.synthetic_captions}'
    else:
        output_name += (f'{config.experiment}_{args.dataset}' + (f'_{args.language}' if args.dataset == 'laion' else '')
                        + (f'_ner' if args.ner else ''))
    return output_name


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser('Train model from zero', add_help=False)
    parser.add_argument('-si', '--synthetic_images', default=False, type=bool)
    parser.add_argument('-sc', '--synthetic_captions', default=False, type=bool)
    parser.add_argument('-d', '--dataset', default='synthetic', type=str)
    parser.add_argument('-e', '--date_estimation', default=False, type=bool)
    parser.add_argument('-w', '--weighted_criterion', default=False, type=bool)
    parser.add_argument('-l', '--language', default=None, type=str)
    parser.add_argument('-n', '--ner', default=False, type=bool)
    parser.add_argument('-t', '--token', default=True, type=bool)
    args = parser.parse_args()

    config = Config()
    main(config, args)
