"""
Stealing from annotated transformer page
"""
import copy
import math
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
    Either an encoder or decoder layer.  Or something else?
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

    def forward(self, x, kvs=None, masks=None):

        # First attend to self, then others
        # Definitely in premature generalizing territory here, but this
        # actually looks like it could lead somewhere pretty interesting
        if not kvs:
            kvs = [(x, x)] * (len(self.attns) - 1)

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

    def forward(self, x, masks, kvs=None):
        for layer in self.layers:
            x = layer(x, masks, kvs)
        x = self.norm(x)
        return x


class MultiHeadedAttention(nn.Module):
    """
    I guess this is where the magic happens
    """

    def __init__(self, h, d_k, dropout=0.1):
        super().__init__()
        self.h = h
        self.d_k = d_k

        # Easier to just define the model size off one of the sub arrays than
        # go through an assertion song and dance
        d_model = d_k * h
        self.linears = cloner(nn.Linear(d_model, d_model), 4)

        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def do_attention(query, key, value, mask=None, dropout=None):
        """
        Nothing really fancy here.  Going for:
        softmax( (Q * K^T)/sqrt(d_k)) * V
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout:
            p_attn = dropout(p_attn)

        attn = torch.matmul(p_attn, value), p_attn
        return attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # Ugh, just going to trust the annotated page for the dimensions here
        qkv = []
        for f, x in zip(self.linears, (query, key, value)):
            qkv.append(f(x).view(nbatches, -1, self.h,
                                 self.d_k).transpose(1, 2))

        query, key, value = qkv

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # so many dimension shenanigans
        x = x.transpose(1, 2).contiguous()
        x = x.view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        return x


class FF(nn.Module):
    """
    Regular feed forward bit
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.W_1(x))
        x = self.dropout(x)
        x = self.W_2(x)
        return(x)


class Embedder(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model_sqrt = math.sqrt(d_model)

    def forward(self, x):
        x = self.emb(x) * self.d_model_sqrt
        return(x)


class PositionalEncoding(nn.Module):
    """
    This is the weirdo sin/cos positional thing
    Unclear whether it would be useful to look at alternative positional encodings
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        # Sigh at magic number here.  This 1000 came from the blog post.  I guess
        # it's just for keeping the numbers convenient?
        div_term = torch.arange(0, d_model, 2) * -(math.log(1000) / d_model)
        div_term = torch.exp(div_term)

        pos_div = pos * div_term
        pe[:, 0::2] = torch.sin(pos_div)
        pe[:, 1::2] = torch.cos(pos_div)
        pe = pe.unsqueeze(0)

        # TODO: figure out what this does
        self.register_buffer('pe', pe)

    def forward(self, x):
        # TODO: check if requires_grad is still necessary
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        x = self.dropout(x)
        return(x)


class EncoderDecoder(nn.Module):
    pass
