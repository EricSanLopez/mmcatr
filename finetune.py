import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time
import sys

from models import utils, caption
from datasets.dataloader import build_dataloader
from configuration import Config
from engine import train_one_epoch, train_one_epoch_multitask, evaluate, evaluate_multitask

import argparse

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


def finetune(config, args):
    output_name = get_output_name(args)
    with (wandb.init(project="XAC-ImageCaptioning",
                     name=f'Finetune_{output_name}',
                     config=config)):  # Starting wandb
        device = torch.device(config.device)
        print(f'Initializing Device: {device}')

        seed = config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        checkpoint_path = f'/data2fast/users/esanchez/checkpoints/{args.checkpoint}.pth'
        model, criterion = caption.build_model(config, multimodal=(args.date_estimation or args.multimodal_model))

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            raise NotImplementedError('Incorrect checkpoint from the checkpoints folder')
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        # config.lr = 1e-5
        config.epochs = args.epochs
        # config.lr_drop = 8

        criterion_datation = utils.get_criterion("NDCG")
        optimizer, lr_scheduler = utils.get_optimizer(model, config)

        data_loader_train, data_loader_val, aux = build_dataloader(
            args.dataset, args.date_estimation, config, args.language, args.ner, args.synthetic_images,
            args.weighted_criterion, args.token)

        criterion = aux if args.weighted_criterion else criterion

        print("Start Training..")
        global_loss, global_validation_loss = utils.ArrayStructure(multitask=args.date_estimation), \
            utils.ArrayStructure(multitask=args.date_estimation)

        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")

            # Training
            if args.date_estimation or args.multimodal_model:
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
            }, os.path.join('/data2fast/users/esanchez/checkpoints', output_name + '.pth'))

            # Validation
            if args.date_estimation or args.multimodal_model:
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
    output_name = f'checkpoint_from_{args.checkpoint}_to_{args.dataset}'
    if args.dataset == 'xac':
        output_name += f'_ner_{args.ner}'
    elif args.dataset == 'laion':
        output_name += f'_{args.language}'
    elif args.dataset == 'synthetic':
        output_name += f'_{args.synthetic_images}_{args.language}'
    return output_name


if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser('Finetune model', add_help=False)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default='coco', type=str)
    parser.add_argument('--weighted_criterion', default=False, type=bool)
    parser.add_argument('--multimodal_model', default=False, type=bool)
    parser.add_argument('--date_estimation', default=False, type=bool)
    parser.add_argument('--ner', default=False, type=bool)
    parser.add_argument('--language', default=None, type=str)
    parser.add_argument('-si', '--synthetic_images', default=False, type=bool)
    parser.add_argument('--token', default=True, type=bool)
    args = parser.parse_args()

    config = Config()
    finetune(config, args)
