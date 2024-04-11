import torch
from torch import nn
import torch.nn.functional as F

from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer


class Multitask(nn.Module):
    def __init__(self, encoder, decoder, datation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.datation = datation

    def forward(self, task, samples, target=None, target_mask=None):
        memory, mask, pos = self.encoder(samples)
        match task:
            case 'captioning':
                return self.decoder(memory, mask, pos, target, target_mask)
            case 'datation':
                return self.datation(memory)
            case _:
                raise NotImplementedError(f'Task {task} not in ["captioning", "datation"]')


class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask,
                              pos[-1], target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class CaptionEncoder(nn.Module):
    def __init__(self, backbone, encoder, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.encoder = encoder

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        memory = self.encoder(self.input_proj(src), mask, pos[-1])
        return memory, mask, pos


class CaptionDecoder(nn.Module):
    def __init__(self, decoder, hidden_dim, vocab_size):
        super().__init__()
        self.decoder = decoder
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, memory, mask, pos, target, target_mask):
        hs = self.decoder(memory, mask, pos[-1], target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Datation(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        hidden_dim = 512
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.max_pool = nn.MaxPool1d(kernel_size=196)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.max_pool(x)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def build_model(config, multimodal=False):
    backbone = build_backbone(config)
    if not multimodal:
        transformer = build_transformer(config, True)
        model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
        model.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
        model.transformer.embeddings.word_embeddings = nn.Embedding(
            config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
    else:
        encoder, decoder = build_transformer(config)
        cap_encoder = CaptionEncoder(backbone, encoder, config.hidden_dim)
        cap_decoder = CaptionDecoder(decoder, config.hidden_dim, config.vocab_size)
        datation = Datation(output_dim=256)
        model = Multitask(cap_encoder, cap_decoder, datation)
        model.decoder.mlp.layers[2] = nn.Linear(512, config.new_vocab_size, bias=True)
        model.decoder.decoder.embeddings.word_embeddings = nn.Embedding(
            config.new_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion
