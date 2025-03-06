from torch.nn import functional as F
import torch
from torch import nn


class MaxMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.input_dim = cfgs['model']["input_dim"]
        self.embed_dim = cfgs['model']["embed_dim"]
        self.aggregator = nn.Linear(self.input_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, 2)
        # self.fre = FreFeature()
        # self.causal = Intervention()

    def forward(self, x):
        # z = self.fre(x.reshape([1, -1, 512]))
        x = self.aggregator(x)
        # x = self.causal(x)
        y = self.classifier(x)
        return y
    
    def inference(self, x):
        x = x.reshape([-1, self.input_dim])
        logic = self.forward(x)
        _idx = torch.argmax(logic[:, 1])
        prob = torch.softmax(logic[_idx], dim=-1)
        pred = prob.argmax(dim=-1)
        return pred
