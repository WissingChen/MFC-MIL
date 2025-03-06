import torch
from torch import nn
from torch.nn import functional as F
from .layer import CrossTransLayer
import math
import numpy as np

class DownSample(nn.Module):
    def __init__(self, d_model=512, dropout=.1, max_len=60000):
        super().__init__()
        """translate x20 feature to x5
        """
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.proj = nn.Conv2d(d_model, d_model, 7, 1, 7//2, groups=d_model)
        self.proj1 = nn.Conv2d(d_model, d_model, 5, 1, 5//2, groups=d_model)
        self.proj2 = nn.Conv2d(d_model, d_model, 3, 1, 3//2, groups=d_model)
        self.down_sampling_1 = nn.Conv1d(d_model, d_model, 16, 16, 0, 1)
        self.down_sampling_2 = nn.Conv1d(d_model, d_model, 16, 16, 15, 3)
        self.down_sampling_3 = nn.Conv1d(d_model, d_model, 16, 16, 30, 5)
        self.down_sampling_res = nn.MaxPool1d(16, 16, 0, 1)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        x = self.norm(x)
        B, H, C = x.size()
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) #[B, N, 512]
        x = x.transpose(1, 2).view(B, C, _H, _W)
        x = self.proj(x)+self.proj1(x)+self.proj2(x)+x
        x = x.flatten(2).transpose(1, 2)
        x_hl = x[:, :-add_length, :] if add_length > 0 else x
        # x_hl = self.dropout(x)  # [bs, length, d_model]
        x_ll = x_hl.transpose(-1, -2)  # [bs, d_model, length]
        # print(add_length, x.size(), x_ll.size())
        x_ll = self.down_sampling_1(x_ll) + self.down_sampling_2(x_ll) + self.down_sampling_3(x_ll) + self.down_sampling_res(x_ll)# [bs, d_model * 4, length // 16]
        x_ll = x_ll.transpose(-1, -2) # [bs, length // 16, d_model * 4]
        x_ll = F.gelu(x_ll)
        x_ll = self.out(x_ll)
        return x_hl, x_ll
    

class MultiLevelFuse(nn.Module):
    def __init__(self, d_model=512, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.hl_weight = nn.Linear(512, 1)
        self.ll_weight = nn.Linear(512, 1)
    
    def forward(self, x, hl, ll):
        x = self.norm(x)
        x_ll = torch.sum(ll * self.ll(ll), dim=-1, keepdim=True)
        x_hl = torch.sum(hl * self.hl(hl), dim=-1, keepdim=True)

        x = x + x_ll, x_hl
        return x

