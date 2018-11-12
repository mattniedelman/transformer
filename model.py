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


class XcoderLayer(nn.Module):
    """
    Either an encoder or decoder layer
    """

    def __init__(self, nfeatures, attns, feed_forward, dropout):
        super().__init__()
        self.nfeatures = nfeatures
        self.attns = attns
        self.feed_forward = feed_forward

        # Need an amount of sublayers equal to the amount of attention steps + 1
        # For encoder, there is 1 attention step, for decoders, 2.  It's not
        # obvious to me that this couldn't be repeated as desired to fit more
        # complex inputs
        self.sublayers = cloner(SublayerConnection(
            nfeatures, dropout), len(attns) + 1)

    def forward(self, x, kvs=[(x, x)], masks=None, mem=None):

        # Create key value sequence
        kvs = [(x, x)] * len(self.attns)

        # First attend to self, then others
        # Definitely in premature generalizing territory here, but this
        # actually looks like it could lead somewhere pretty interesting
        # In any case, mem is still static here, but a sequence of different
        # and arbitrary kv to query against would be SUPER interesting
        if mem:
            kvs[1:] = [(mem, mem)] * (len(self.attns) - 1)

        # For each attention step, grab a sublayer, perform the attention step
        # with the corresponding key, value, and mask.
        # TODO: it will probably be useful to be more structured about these args
        for layer, attn, kv, mask in zip(self.sublayers, self.attns, kvs, masks):
            k, v = kv
            x = layer(x, lambda x: attn(query=x,
                                        key=k,
                                        value=v,
                                        mask=mask))

        x = self.sublayers(x, self.feed_forward)
        return x


class Xcoder(nn.Module):
    """
    Generic class, either encoder or decoder
    The only difference is the presence of additional memory inputs
    """

    def __init__(self, layer, nlayers):
        super().__init__()
        self.layers = cloner(layer, nlayers)
        self.norm = LayerNorm(layer.nfeatures)

    def forward(self, x, masks, mem=None):
        for layer in self.layers:
            x = layer(x, masks, mem)
        x = self.norm(x)


class EncoderDecoder(nn.Module):
    pass
