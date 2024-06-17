import os
import annoy
import torch
import warnings
import numpy as np
import pandas as pd
import torch.multiprocessing

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


class Annoyer:
    # High performance approximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, model, dataset, emb_size=None, distance='angular', experiment_name='resnet_base',
                 out_dir='output/', device='cuda') -> None:
        assert not (emb_size is None) and isinstance(emb_size, int),\
            f'When using Annoyer KNN emb_size must be an int. Set as None for common interface. Found: {type(emb_size)}'

        self.model = model
        self.model.eval()

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataloader = dataset
        self.device = device
        self.anys = dict()

        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(
            out_dir, f'index.ann')
        self.year_path = os.path.join(
            out_dir, 'year_index.csv'
        )

        self.trees = annoy.AnnoyIndex(emb_size, distance)
        self.state_variables = {
            'built': False,
        }

    def fit(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot fit a built Annoy')
        else:
            self.state_variables['built'] = True

        for idx, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            tasks, images, masks, years = data
            samples = [images.to('cuda'), masks.to('cuda')]

            with torch.no_grad():
                embeddings = self.model.forward(tasks[0], samples).squeeze()  # Ensure batch_size = 1

            for i, ret in enumerate(zip(embeddings, years)):
                emb, year = ret
                self.trees.add_item(idx*len(tasks) + i, emb)
                self.anys[idx*len(tasks) + i] = year.item()

        self.trees.build(10)  # 10 trees
        self.trees.save(self.path)
        pd.DataFrame(self.anys.items(), columns=['index', 'year']).to_csv(self.year_path, index=False)

    def load(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot load an already built Annoy')
        else:
            self.state_variables['built'] = True

        self.trees.load(self.path)
        df = pd.read_csv(self.year_path)
        self.anys = {k: v for k, v in zip(df['index'], df['year'])}

    def retrieve_by_idx(self, idx, n=50, **kwargs):
        return self.trees.get_nns_by_item(idx, n, **kwargs)

    def retrieve_by_vector(self, vector, n=50, **kwargs):
        idxs = self.trees.get_nns_by_vector(vector, n, **kwargs)
        return np.array([self.anys[idx] for idx in idxs])
