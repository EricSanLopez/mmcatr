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
from datasets.dataloader import build_dataloader
from datasets.utils import read_json, tkn
from configuration import Config
from engine import train_one_epoch, evaluate

import argparse

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


def validate(config, args):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.checkpoint is None:
        checkpoints = [f'experiment_{si}_{sc}_to_xac_ner_{args.ner}' for si, sc in
                       zip([True, True, False, False], [True, False, True, False])]
    else:
        checkpoints = [args.checkpoint]

    if args.loss:
        handler = open('validation.txt', 'w')
        handler.write("checkpoint\tvalidation_loss\n")

    data_loader_train, data_loader_val, criterion = build_dataloader(
        args.dataset, False, config, args.language, args.ner, False,
        False, True, False)

    langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
    special_tokens_dict = {'additional_special_tokens': tkn(langs)}
    tokenizer.add_tokens(additional_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)

    for experiment in checkpoints:
        model, _ = caption.build_model(config)
        checkpoint_path = f'checkpoints/{experiment}.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except:
            raise NotImplementedError('Incorrect checkpoint from coco, hist_sd or xac')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        if args.loss:
            print(f"Start Validation for {experiment}")
            validation_loss = evaluate(model, criterion, data_loader_val, device)
            handler.write(f"{experiment}\t{validation_loss}\n")
            print(f"Validation Loss: {validation_loss}")

        print(f"Start predicts..")
        preds, targets = predict(model, data_loader_val, tokenizer, config)
        preds = [tokenizer.decode(pred.tolist(), skip_special_tokens=True).capitalize() for pred in preds]
        targets = [[tokenizer.decode(target.tolist(), skip_special_tokens=True).capitalize()] for target in targets]
        handler_aux = open(f'logs/examples_{experiment}.txt', 'w')
        handler_aux.write("prediction\ttarget\n")
        handler_aux.write('\n'.join(['\t'.join([p, t[0]]) for p, t in zip(preds[:50], targets[:50])]))
        handler_aux.close()

        print("Start NLP metrics..")
        eval_df = eval_nlp_metrics(preds, targets)
        eval_df.to_csv(f'logs/nlp_metrics_{experiment}.csv', index=False)
        print(f"NLP metrics: \n{eval_df}")

    if args.loss:
        handler.close()
    return


@torch.no_grad()
def predict(model, data_loader, tokenizer, config, device='cuda'):
    preds = list()
    targets = list()

    total = len(data_loader)
    with tqdm.tqdm(total=total) as pbar:
        for tasks, images, masks, caps, cap_masks in data_loader:
            images = images.to(device)

            start_token = caps[0][0].item()
            end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
            captions = torch.zeros((caps.shape[0], config.max_position_embeddings), dtype=torch.long, device=device)
            captions_masks = torch.ones((caps.shape[0], config.max_position_embeddings),
                                        dtype=torch.bool, device=device)

            captions[:, 0] = start_token
            captions_masks[:, 0] = False

            for i in range(config.max_position_embeddings - 1):
                predictions = model(images, captions, captions_masks)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, dim=1)

                # Check if any sentence is finished
                pred_list = predicted_id.detach().cpu().numpy()
                if not all(pred_list - end_token):
                    idx = np.argwhere(pred_list == end_token).flatten()
                    preds.extend(captions[idx])
                    targets.extend(caps[idx])

                    finished_mask = np.ones(len(pred_list), bool)
                    finished_mask[idx] = False
                    captions = captions[finished_mask]
                    captions_masks = captions_masks[finished_mask]
                    images = images[finished_mask]
                    caps = caps[finished_mask]
                    predicted_id = predicted_id[finished_mask]

                    if not any(finished_mask):
                        # print(f'########################### {i} #################################')
                        break

                captions[:, i + 1] = predicted_id
                captions_masks[:, i + 1] = False

                # print(captions[0])

            else:
                # print('########################### 128 #################################')
                print(tokenizer.decode(captions[0].tolist(), skip_special_tokens=True).capitalize())
                preds.extend(captions)
                targets.extend(caps)

            pbar.update(1)
    return preds, targets


def eval_nlp_metrics(preds, caps):
    caps = [c if len(c[0]) > 0 else ['<unk>'] for c in caps]

    evaluate = Evaluate(metrics=["cider_d", "bleu"])
    corpus_scores, _ = evaluate(preds, caps)

    corpus_scores = {k: [v.item()] for k, v in corpus_scores.items()}
    return pd.DataFrame().from_dict(corpus_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Validate model', add_help=False)
    parser.add_argument('--dataset', default='xac', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--loss', default=False, type=bool)
    parser.add_argument('--ner', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--language', default=None, type=str)
    args = parser.parse_args()

    config = Config()
    validate(config, args)
