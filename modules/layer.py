
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import math
from einops import rearrange, reduce
from math import ceil
from timm.models.vision_transformer import Block


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z




class CrossNystromAttention(NystromAttention):
    def __init__(self, 
                dim,
                dim_head = 64,
                heads = 8,
                num_landmarks = 256,
                pinv_iterations = 6,
                residual = True,
                residual_conv_kernel = 33,
                eps = 1e-8,
                dropout = 0.
        ):
        super(NystromAttention, self).__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)
        self.to_qkv = nn.ModuleList([
            nn.Linear(dim, inner_dim, bias = False),
            nn.Linear(dim, inner_dim * 2, bias = False)
            ])

    def forward(self, x, z, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        n_z = z.size(1)

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

        remainder_z = n_z % m
        if remainder_z > 0:
            padding_z = m - (n_z % m)
            z = F.pad(z, (0, 0, padding_z, 0), value = 0)

            # if exists(mask):
                # mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q = self.to_qkv[0](x)# .chunk(3, dim = -1)
        k, v = self.to_qkv[1](z).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k, v = map(lambda t: rearrange(t, 'b n_z (h d) -> b h n_z d', h = h), (k, v))

        # set masked positions to 0 in queries, keys, values

        # if exists(mask):
        #     mask = rearrange(mask, 'b n -> b () n')
        #     q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        l_z = ceil(n_z / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        landmark_einops_eq_z = '... (n_z l_z) d -> ... n_z d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq_z, 'sum', l_z = l_z)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        # if exists(mask):
            # mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            # divisor = mask_landmarks_sum[..., None] + eps
            # mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / l_z

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        # if exists(mask):
        #     mask_value = -torch.finfo(q.dtype).max
        #     sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
        #     sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
        #     sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class CrossTransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.cross_attn = CrossNystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, z):
        x = x + self.cross_attn(x, z)
        return x

