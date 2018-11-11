"""
Stealing from annotated transformer page
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tourch.autograd import Variable


class SublayerConnection(nn.Module):
    """
    One of the components of a layer
    A res connection + layer norm
    """
    pass


class LayerNorm(nn.Module):
    """
    Other component of the layers
    TODO: maybe should be combined with the sublayer connection class?
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = (x-mean) / (std + self.eps)
        normed = self.w * norm + self.b
        return normed


class EncoderLayer(nn.Module):
    pass


class DecoderLayer(nn.Module):
    pass


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class EncoderDecoder(nn.Module):
    pass
