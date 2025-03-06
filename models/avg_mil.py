import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F
# from models.fremil import Intervention


class AvgMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.input_dim = cfgs['model']["input_dim"]
        self.embed_dim = cfgs['model']["embed_dim"]
        self.aggregator = nn.Linear(self.input_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, cfgs['model']["num_classes"])
        # self.causal = Intervention()
        
    def forward(self, x):
        x = self.aggregator(x)
        # x = self.causal(x)
        y = self.classifier(x)
        return y
    
    def inference(self, x):
        x = x.reshape([-1, self.input_dim])
        logic = self.forward(x)
        # _idx = torch.argsort(logic[:, 1], descending=True)[:self.cfgs["dataset"]["k"]]
        # _logic = logic[_idx]
        logic = logic.mean(dim=0)
        prob = F.softmax(logic, dim=-1)
        pred = prob.argmax(dim=-1)
        return pred
