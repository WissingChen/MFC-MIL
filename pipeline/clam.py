from .transmil import TransMILPipeline
import numpy as np
import torch
from torch.nn import functional as F
import time
import torch
from utils import build_lr_scheduler


class CLAMPipeline(TransMILPipeline):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.lr_scheduler = build_lr_scheduler(cfgs, self.optimizer, len(self.train_dataloader))

    def _train_iter(self, epoch):
        loss = self._get_loss()

        self.monitor.log_info(f'\n[train] Epoch: [{epoch}/{self.cfgs["optim"]["epochs"]}]\tLoss: {loss}\n')
    
    def _get_loss(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_dataloader):
            # [1*k, N] -> [B*K, N]
            feature = feature.cuda().float()
            target = target.cuda()
            output, _, _, _, instance_dict = self.model(feature, target, instance_eval=True)

            loss_bag = self.criterion(output, target)
            instance_loss = instance_dict['instance_loss']
        
            loss = self.cfgs['optim']["bag_weight"] * loss_bag + (1-self.cfgs['optim']["bag_weight"]) * instance_loss 
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print(f"\r[train] backward progress: {(i+1)/len(self.train_dataloader):.1%}", end='')
            self.lr_scheduler.step()
        return running_loss/len(self.train_dataloader)
    
    def inference(self, loader):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                feature = feature.cuda().float()
                output  = self.model(feature)[1]
                probs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                print(f'\r[test] inference progress: {i+1}/{len(loader)}', end='')
        return probs, targets
