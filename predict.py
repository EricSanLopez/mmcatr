import torch
import torch.nn as nn

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
from datasets.utils import read_json, tkn

import warnings
warnings.filterwarnings("ignore")  # Ignoring bleu score unnecessary warnings.


parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='None')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
        raise NotImplementedError('No model to chose from!')
    else:
        if not os.path.exists(checkpoint_path):
            raise NotImplementedError('Give valid checkpoint path')
        print("Found checkpoint! Loading!")
        model, _ = caption.build_model(config)
        model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
        model.transformer.embeddings.word_embeddings = nn.Embedding(
            config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        print("Loading Checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
special_tokens_dict = {'additional_special_tokens': tkn(langs)}
tokenizer.add_tokens(additional_tokens)
tokenizer.add_special_tokens(special_tokens_dict)

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == end_token:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


output = evaluate()
print(output[0].tolist())
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result.capitalize())
