from .transmil import TransMILPipeline
import numpy as np
import torch
from torch.nn import functional as F
import time
import torch
from utils import build_lr_scheduler
from cluster import reduce


class DSMILPipeline(TransMILPipeline):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.lr_scheduler = build_lr_scheduler(cfgs, self.optimizer, len(self.train_dataloader))

    def _train_iter(self, epoch):
        loss = self._get_loss()

        self.monitor.log_info(f'\n[train] Epoch: [{epoch}/{self.cfgs["optim"]["epochs"]}]\tLoss: {loss}\n')
    
    def _get_loss(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, _target, slide_id) in enumerate(self.train_dataloader):
            # [1*k, N] -> [B*K, N]
            feature = feature.cuda().float()
            target = torch.tensor([0., 1.]).cuda() if _target == 1 else torch.tensor([1., 0.]).cuda()
            output = self.model(feature)
            max_prediction, index = torch.max(output["i_logits"], 0)
            loss = self.criterion(output['logits'].view(1, -1), target.view(1, -1)) + self.criterion(max_prediction.view(1, -1), target.view(1, -1))
            loss = loss / 2.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print(f"\r[train] backward progress: {(i+1)/len(self.train_dataloader):.1%}", end='')
            self.lr_scheduler.step()
        return running_loss/len(self.train_dataloader)
