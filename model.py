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
    Layernorm, for use in sublayer connections
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
    Use layernorm class to get some residual connections going with an
    arbitrary sublayer.  Looks like it's always self-attention + feed forward
    in transformer
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
    Compose the self-attention and feed forward layers using the sublayer
    connection class
    """

    def __init__(self, nfeatures, self_attn, feed_forward, dropout):
        super().__init__()
        self.nfeatures = nfeatures
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = cloner(SublayerConnection(nfeatures, dropout), 2)

    def forward(self, x, mask):
        attn, ff = self.sublayers
        # TODO: what's going on with the repeated x here
        attended = attn(x, lambda x: self.self_attn(x, x, x, mask))
        ffed = ff(attn, self.feed_forward)
        return ffed


class Encoder(nn.Module):
    """
    Stack up a bunch of encoder layers
    """

    def __init__(self, layer, nlayers):
        super().__init__()
        self.layers = cloner(layer, nlayers)
        self.norm = LayerNorm(layer.nfeatures)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    """
    This is almost exactly the same as the encoder layers
    Will come back later and merge them
    """

    def __init__(self, nfeatures, self_attn, feed_forward, dropout, src_attn):
        super().__init__()
        self.nfeatures = nfeatures
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = cloner(SublayerConnection(nfeatures, dropout), 2)

    def forward(self, x, mem, src_mask, tgt_mask):
        attn_self, attn_src, ff = self.sublayers

        x = attn_self(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = attn_src(x, lambda x: self.src_attn(x, mem, mem, src_mask))
        x = ff(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    Also pretty much the same as the encoder
    """

    def __init__(self, layer, nlayers):
        super().__init__()
        self.layers = cloner(layer, nlayers)
        self.norm = LayerNorm(layer.nfeatures): w

    def forward(self, x, mem, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)

        x = self.norm(x)
        return(x)


class EncoderDecoder(nn.Module):
    pass
