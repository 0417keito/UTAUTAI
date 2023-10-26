import math

import torch
from torch import nn

from einops import rearrange

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# t5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale = 1.,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n):
        pos = torch.arange(n, device = self.device).long()
        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')

        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)

        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale
