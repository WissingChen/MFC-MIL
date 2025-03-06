import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import math
from einops import rearrange, reduce
from math import ceil
from timm.models.vision_transformer import Block

class AF(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(AF, self).__init__()
        """
        Attention Fusion Module
        """
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(.1)

    def forward(self, q, k, v, proj=False):
        if proj:
            qk = torch.matmul(self.q(q), self.k(k).transpose(-1, -2)) / math.sqrt(q.size(-1))
            score = qk.softmax(dim=-1)
            score = self.dropout(score)
            out = score.matmul(self.v(v))
        else:
            qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
            score = qk.softmax(dim=-1)
            out = score.matmul(v)
        return out


class FDIntervention(nn.Module):
    def __init__(self, embed_dim=512):
        super(FDIntervention, self).__init__()
        self.embed_dim = embed_dim
        self.af_1 = AF(embed_dim)
        self.af_2 = AF(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, feature, mediator, proj=False):
        v = self.af_1(mediator, feature, feature, proj)
        out = self.af_2(feature, mediator, v, proj)
        out = self.fc(out + feature)
        return out


class BDIntervention(nn.Module):
    def __init__(self, input_dim, embed_dim=1024):
        super(BDIntervention, self).__init__()
        self.af = AF(input_dim, embed_dim)
    
    def forward(self, feature, confounder, proj=False):
        out = self.af(feature, confounder, confounder, proj)
        # out = self.norm(out + feature)
        return out



class InstanceMemory(nn.Module):
    def __init__(self, memory_size=16, d_model=512):
        super(InstanceMemory, self).__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        # slot memory
        self.memory = nn.Parameter(torch.randn(memory_size, d_model)).cuda()
        self.write_weight = nn.Linear(d_model, memory_size)
        self.read_weight = nn.Linear(d_model, memory_size)

    def forward(self, x):
        # x [1, L, D]
        # Write to memory
        x = x.reshape([-1, self.d_model])
        write_weight = F.softmax(self.write_weight(x), dim=-1) # [L, MS]
        memory = self.memory + torch.matmul(write_weight.T, x) # [MS, D]
        
        # Read from memory
        read_weight = F.softmax(self.read_weight(x), dim=-1)
        read_vector = torch.matmul(read_weight, memory)
        
        return read_vector.unsqueeze(0)


class InterventionOnline(nn.Module):
    def __init__(self, input_dim=512, embed_dim=512, k=32):
        super().__init__()
        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)

        self.out = nn.Linear(embed_dim * 2, input_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
    
    def forward(self, x, x_hat, res=False):
        x = self.norm(x)
        x = x.reshape([1, -1, self.input_dim])
        q = self.q(x)
        k = self.k(x_hat)

        qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        score = qk.softmax(dim=-1)
        out = self.v(score.matmul(k))

        out = self.out(torch.cat([q, out], dim=-1))
        if res:
            return out
        else:
            return out + x
