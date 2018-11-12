"""
Stealing from annotated transformer page
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tourch.autograd import Variable


def cloner(module, nclones):
    """
    Make a bunch of clones of a layer
    """
    clonelist = [copy.deepcopy(module) for _ in range(nclones)]
    return nn.ModuleList(clonelist)


class LayerNorm(nn.Module):
    """
    Other component of the layers
    TODO: maybe should be combined with the sublayer connection class?
    """

    def __init__(self, nfeatures, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(nfeatures))
        self.b = nn.Parameter(torch.zeros(nfeatures))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = (x - mean) / (std + self.eps)
        normed = self.w * norm + self.b
        return normed


class SublayerConnection(nn.Module):
    """
    One of the components of a layer
    A res connection + layer norm
    """

    def __init__(self, nfeatures, dropout):
        super().__init__()
        self.norm = LayerNorm(nfeatures)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        normed = self.norm(x)
        out = x + self.dropout(sublayer(normed))
        return out


class EncoderLayer(nn.Module):
    """
    """

    def __init__(self, nfeatures, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(nfeatures, dropout), 2)

    def forward(self, x, mask):
        attn, ff = self.sublayer
        # TODO: what's going on with the repeated x here
        attended = attn(x, lambda x: self.self_attn(x, x, x, mask))
        ffed = ff(attn, self.feed_forward)
        return ffed


class DecoderLayer(nn.Module):
    pass


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class EncoderDecoder(nn.Module):
    pass
