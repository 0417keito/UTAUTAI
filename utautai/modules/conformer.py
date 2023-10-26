from torch import nn
from einops.layers.torch import Rearrange

from ..utils.utils import calc_same_padding, exists
from .modules import (GLU, DepthWiseConv1d, Swish, ChanLayerNorm, FeedForward,
                      Attention, PreNorm, Scale)
from .positional_embedding import RotaryEmbedding, T5RelativePositionBias


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        attn_bias = None
    ):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True,
        t5_rel_pos_bias = False
    ):
        super().__init__()

        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        self.dim = dim
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash
            ))

    def forward(self, x, mask = None):
        seq_len = x.shape[-2]

        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None

        for block in self.layers:
            x = block(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias
            )

        return x
