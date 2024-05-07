# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional
from collections.abc import Callable
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score

from annoy_index import annoy

from pytorch_metric_learning import miners, losses


class ArrayStructure:
    def __init__(self, array=None, multitask=False):
        if array is None:
            self.array = dict() if multitask else list()
            self.multitask = multitask
        else:
            self.array = array
            self.multitask = type(array) == dict

    def __add__(self, values):
        if self.multitask:
            for task in values.keys():
                try:
                    self.array[task].append(values[task])
                except KeyError:
                    self.array[task] = [values[task]]
            return ArrayStructure(self.array)
        else:
            self.array.append(values)
            return ArrayStructure(self.array)


def save_losses(output_name, epoch_loss, validation_loss):
    with open(os.path.join('checkpoints', output_name + '.csv'), 'w') as f:
        if epoch_loss.multitask:
            f.write(f'epoch,{",".join([f"training_{task}" for task in epoch_loss.array.keys()])},'
                    f'{",".join([f"validation_{task}" for task in validation_loss.array.keys()])}\n')
            [f.write(f'{i+1},{",".join([str(epoch_loss.array[task][i]) for task in epoch_loss.array.keys()])},'
                     f'{",".join([str(validation_loss.array[task][i]) for task in validation_loss.array.keys()])}\n')
             for i in range(len(epoch_loss.array.values()))]
        else:
            f.write('epoch,training,validation\n')
            f.write('\n'.join([f'{i + 1},{losses[0]},{losses[1]}' for i, losses in
                               enumerate(zip(epoch_loss.array, validation_loss.array))]))


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    elif len(tensor_list) == 2:
        return NestedTensor(tensor_list[0], tensor_list[1])
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def sigmoid(x, k=1.0):
    exponent = -x/k
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1./(1. + torch.exp(exponent))
    return y


class CosineSimilarityMatrix(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityMatrix, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return cosine_similarity_matrix(x1, x2, self.dim, self.eps)


def cosine_similarity_matrix(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    '''
    When using cosine similarity the constant value must be positive
    '''
    #Cosine sim:
    xn1, xn2 = torch.norm(x1, dim=dim), torch.norm(x2, dim=dim)
    x1 = x1 / torch.clamp(xn1, min=eps).unsqueeze(dim)
    x2 = x2 / torch.clamp(xn2, min=eps).unsqueeze(dim)
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(1)

    sim = torch.tensordot(x1, x2, dims=([2], [2])).squeeze()

    sim = (sim + 1)/2 #range: [-1, 1] -> [0, 2] -> [0, 1]

    return sim


def ndcg(output, target, reducefn='mean'):
    # Similarity matrix
    sm = cosine_similarity_matrix(output, output)
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    gt = torch.abs(target.unsqueeze(0) - target.unsqueeze(1))
    gt = gt[mask_diagonal].view(sm.shape[0], sm.shape[0]-1).float()

#    relevance = 1. / (gt + 1)
#    relevance = 10-gt
    relevance = 5-gt
    relevance = relevance.clamp(max=5, min=0)
    relevance = relevance.exp() - 1

    ndcg_sk = []
    for y_gt, y_scores in zip(relevance, ranking):
        y_scores_np = np.asarray([y_scores.cpu().numpy()])
        y_gt_np = np.asarray([y_gt.cpu().numpy()])
        ndcg_sk.append(ndcg_score(y_gt_np, y_scores_np))

    if reducefn == 'mean':
        return np.mean(ndcg_sk)
    elif reducefn == 'sum':
        return np.sum(ndcg_sk)
    elif reducefn == 'none':
        return ndcg_sk


def year_dif(output, target, reducefn='mean'):
    # Similarity matrix
    sm = cosine_similarity_matrix(output, output)
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0] - 1)

    preds_idx = torch.argmax(ranking, dim=1).type(torch.uint8).tolist()
    preds = target[preds_idx]
    dif = torch.abs(target - preds).float().detach().cpu().numpy()

    if reducefn == 'mean':
        return np.mean(dif)
    elif reducefn == 'sum':
        return np.sum(dif)
    elif reducefn == 'none':
        return dif


class DGCLoss(nn.Module):
    def __init__(self, k: float = 1e-3, penalize=False, normalize=True, indicator_function: Callable = sigmoid):
        super(DGCLoss, self).__init__()
        self.k = k
        self.penalize = penalize
        self.normalize = normalize
        self.indicator_function = indicator_function

    def forward(self, ranking: Tensor, gt: Tensor, mask_diagonal: Tensor = None) -> Tensor:
        return dgc_loss(ranking, gt, mask_diagonal=mask_diagonal, k=self.k, penalize=self.penalize,
                        normalize=self.normalize, indicator_function=self.indicator_function)


def dgc_loss(input: Tensor, target: Tensor, mask_diagonal: Tensor = None, k: float = 1e-2, penalize: bool = False,
             normalize: bool = True, indicator_function: Callable = sigmoid,
             similarity: Callable = CosineSimilarityMatrix()) -> Tensor:
    sm = similarity(input, input)
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0] - 1)

    # Ground-truth Ranking function
    gt = torch.abs(target.unsqueeze(0) - target.unsqueeze(1)).float()
    # gt = gt[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    if mask_diagonal is not None:
        # ranking = ranking[mask_diagonal].view(ranking.shape[0], ranking.shape[0]-1)
        gt = gt[mask_diagonal].view(gt.shape[0], gt.shape[0] - 1)

    # Prepare indicator function
    dij = ranking.unsqueeze(1) - ranking.unsqueeze(-1)
    mask_diagonal = ~ torch.eye(dij.shape[-1]).bool()
    dij = dij[:, mask_diagonal].view(dij.shape[0], dij.shape[1], -1)

    # Indicator function
    # Assuming a perfect step function
    # indicator = (dij > 0).float()
    # indicator = indicator.sum(-1) + 1

    # Smooth indicator function
    indicator = indicator_function(dij, k=k)
    indicator = indicator.sum(-1) + 1

    # Relevance score
    #    relevance = 10. / (gt + 1)
    relevance = 10 - gt
    relevance = relevance.clamp(0)
    relevance = relevance.exp2() - 1  # Exponentially penalize
    # Ground-truth Ranking function
    # print(relevance[0])
    if penalize:
        relevance = relevance.exp2() - 1
    # print(relevance.shape, indicator.shape)

    dcg = torch.sum(relevance / torch.log2(indicator + 1), dim=1)

    if not normalize:
        return -dcg.mean()

    relevance, _ = relevance.sort(descending=True)
    indicator = torch.arange(relevance.shape[-1], dtype=torch.float32, device=relevance.device)
    idcg = torch.sum(relevance / torch.log2(indicator + 2), dim=-1)

    dcg = dcg[idcg != 0]
    idcg = idcg[idcg != 0]

    ndcg = dcg / idcg

    if ndcg.numel() == 0:
        return torch.tensor(1, device=ranking.device)
    # Ground-truth Ranking function

    return 1 - ndcg.mean()


class MetricLearning:
    def __init__(self):
        self.miner = miners.BatchEasyHardMiner()
        self.loss_fn = losses.TripletMarginLoss(margin=0.2)

    def __call__(self, outputs, target):
        miner_output = self.miner(outputs, target)
        return self.loss_fn(outputs, target, miner_output)


def get_criterion(criterion):
    match criterion.lower():
        case "ndcg":
            return DGCLoss()
        case "triplet":
            return MetricLearning()
        case other:
            raise NotImplementedError(f'Criterion {criterion} not in ["NDCG", "Triplet"]')


def get_optimizer(model, config):
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
    return optimizer, lr_scheduler
