# Deployment app
import streamlit as st
import torch
import torchvision as tv
from transformers import BertTokenizer

# Import of the model structure
from models import utils, caption
from datasets import coco, synthetic, xac
from datasets.utils import read_json, tkn
from configuration import Config

# Data manipulation
import numpy as np
import torchvision.transforms as T

# Visualization
import matplotlib.pyplot as plt
from PIL import Image

# Other useful libraries
import sys
import os

st.set_page_config(
    page_title="Etiquetador d'imatges d'arxiu",
    layout="wide"
)

config = Config()

# Initialization
device = torch.device(config.device)
print(f'Initializing Device: {device}')

seed = config.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

# Variables and paths
checkpoint_path = "checkpoints/checkpoint_from_experiment_True_True_to_xac_ner_True.pth"
dataset_val = xac.build_dataset(config, ner=True, mode='validation')

langs = read_json(os.path.join("/data2fast/users/esanchez", "laion", 'language-codes.json'))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
additional_tokens = ['<loc>', '<per>', '<org>', '<misc>']
additional_tokens_dict = {'additional_tokens': additional_tokens}
special_tokens_dict = {'additional_special_tokens': tkn(langs)}
tokenizer.add_tokens(additional_tokens_dict)
tokenizer.add_special_tokens(special_tokens_dict)

# Model import
model, criterion = caption.build_model(config)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Predict function
@torch.no_grad()
def predict(image):
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    max_length = 128
    caption = torch.zeros((1, max_length), dtype=torch.long).to(device)
    cap_mask = torch.ones((1, max_length), dtype=torch.bool).to(device)

    caption[:, 0] = start_token
    cap_mask[:, 0] = False

    for i in range(max_length - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == end_token:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


# Page format
st.title("Etiquetador d'imatges d'arxiu")

with st.sidebar:
    st.header('Demo del model generador de descripcions amb dades de la XAC')
    st.markdown("\nModel creat per Èric Sánchez López dins d'una col·laboració del Centre de Visió per Computador amb "
                "la Xarxa d'Arxius Comarcals.")
    st.markdown("Per més informació sobre el model dirigiu-vos al repositori de github.")
    # st.image('../github-mark/github-mark/github-mark.png', width=50)
    st.link_button("Github repository", "https://github.com/EricSanLopez/XACImageCaptioning")

col1, col2 = st.columns((3, 2))

with col1:
    st.header('Insertar fotografia')

    # Transforming the image to the correct format
    file = st.file_uploader("Selecciona l'arxiu en format jpg", type='jpg')
    if file is not None:
        img = Image.open(file)
        st.image(img)
        img = img.convert("RGB")
        img = xac.val_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

with col2:
    st.header('Descripció generada')
    st.markdown('#')
    st.markdown('#')

    if file is None:
        st.button('Generar', type='primary', disabled=True)
    else:
        if st.button('Generar', type='primary'):
            # Feeding the model the image
            with st.spinner("Generant..."):
                cap = predict(img)
                result = tokenizer.decode(cap[0].tolist(), skip_special_tokens=True)
            # Showing the caption and the image
            st.markdown('#### ' + result.capitalize())
