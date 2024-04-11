# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import wandb
import torch

import math
import sys
import tqdm

from models import utils


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for i, data in enumerate(data_loader):
            tasks, images, masks, caps, cap_masks = data
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            if (i + 1) % 50 == 0:
                train_log(loss_value, 'captioning', data[1].shape[0] * i, epoch)

            pbar.update(1)

    return epoch_loss / total


def train_one_epoch_multitask(model, criterion, criterion_datation, data_loader,
                              optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = {'captioning': 0.0, 'datation': 0.0}
    idx = {'captioning': 0, 'datation': 0}
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for data in iter(data_loader):
            task = data[0][0]

            match task:
                case 'captioning':
                    loss = train_captioning(model, data, criterion, device)

                case 'datation':
                    loss = train_datation(model, data, criterion_datation, device)

                case other:
                    raise NotImplementedError(f'Task {task} not in ["captioning", "datation"]')

            loss_value = loss.item()
            epoch_loss[task] += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

            if (idx[task] + 1) % 50 == 0:
                train_log(loss_value, task, data[1].shape[0] * idx[task], epoch)

            idx[task] += 1

    return {k: v / idx[k] for k, v in epoch_loss.items() if idx[k] > 0}


def train_captioning(model, data, criterion, device="cuda"):
    tasks, images, masks, caps, cap_masks = data

    samples = utils.NestedTensor(images, masks).to(device)
    caps = caps.to(device)
    cap_masks = cap_masks.to(device)

    outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])  # No està generalitzat pel cas de només captioning
    loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
    return loss


def train_datation(model, data, criterion, device="cuda"):
    tasks, images, masks, targets = data

    samples = utils.NestedTensor(images, masks).to(device)

    outputs = model(tasks[0], samples)
    targets = (targets - 1930) // 5
    targets = targets.to(device)
    loss = criterion(outputs, targets)
    return loss


def train_log(loss, task, example_ct, epoch):
    """
    Logs on wandb and console.
    """
    print(f"{task} loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    wandb.log({'epoch': epoch, f'loss_{task}': loss})


@torch.no_grad()
def evaluate(model, criterion, criterion_datation, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = {'captioning': 0.0, 'datation': 0.0}
    idx = {'captioning': 0, 'datation': 0}
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for data in iter(data_loader):
            task = data[0][0]

            match task:
                case 'captioning':
                    loss = train_captioning(model, data, criterion, device)

                case 'datation':
                    loss = train_datation(model, data, criterion_datation, device)

                case other:
                    raise NotImplementedError(f'Task {task} not in ["captioning", "datation"]')

            validation_loss[task] += loss.item()
            idx[task] += 1

            pbar.update(1)
        
    return {k: v / idx[k] for k, v in validation_loss.items() if idx[k] > 0}
