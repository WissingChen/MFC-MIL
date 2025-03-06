from .base import BasePipeline
import numpy as np
import torch
from torch.nn import functional as F
import time
import torch


class PoolingMILPipeline(BasePipeline):
    def __init__(self, cfgs):
        super().__init__(cfgs)

    def _train_iter(self, epoch):
        self.train_dataloader.dataset.set_mode('bag')
        probs = self._inference_for_selection(self.train_dataloader)
        self.train_dataloader.dataset.select_instance(probs)
        self.train_dataloader.dataset.set_mode('instance')
        loss = self._get_loss()

        self.monitor.log_info(f'\n[train] Epoch: [{epoch}/{self.cfgs["optim"]["epochs"]}]\tLoss: {loss}\n')
    
    def _get_loss(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, target) in enumerate(self.train_dataloader):
            # [1*k, N] -> [B*K, N]
            input = feature[0].cuda()
            target = target[0].cuda()
            output = self.model(input)

            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print(f"\r[train] backward progress: {(i+1)/len(self.train_dataloader):.1%}", end='')
            self.lr_scheduler.step()
        return running_loss/len(self.train_dataloader)
    
    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = {}
        # slide_name = []
        with torch.no_grad():
            for i, (feature, target, _slide_name) in enumerate(loader):
                input = feature.cuda()
                output = self.model(input)  # [B, num_classes]
                probs[_slide_name[0]] = output.detach().cpu().numpy().reshape([-1, self.cfgs["model"]["num_classes"]])
                # slide_name.extend(_slide_name)
                print(f'\r[train] inference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            # probs = np.array(probs)# .reshape([-1, self.cfgs["model"]["num_classes"]])
        return probs

    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, _slide_name) in enumerate(loader):
                input = feature.cuda()
                output = self.model.inference(input)
                probs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                print(f'\r[test] inference progress: {i+1}/{len(loader)}', end='')
        return probs, targets

    def _train_epoch(self, epoch):
        # loop throuh epochs
        self._train_iter(epoch)
        # Validation
        val_pred, val_target = self.inference(self.val_dataloader)
        val_score = self.metric(val_target, val_pred)
        test_pred, test_target = self.inference(self.test_dataloader)
        test_score = self.metric(test_target, test_pred)

        info = f"""
        Epoch: [{epoch}/{self.cfgs['optim']['epochs']}]
            [ Val ]            [ Test  ]
        Acc: {val_score['Acc.']: .2%}      Acc: {test_score['Acc.']: .2%}          
        AUC: {val_score['AUC']: .2%}      AUC: {test_score['AUC']: .2%}
        Pre: {val_score['Pre.']: .2%}      Pre: {test_score['Pre.']: .2%}          
        Rec: {val_score['Rec.']: .2%}      Rec: {test_score['Rec.']: .2%}          
        F1:  {val_score['F1']: .2%}       F1:  {test_score['F1']: .2%}
        Spe: {val_score['Spe.']: .2%}      Spe: {test_score['Spe.']: .2%}          
"""
        self.monitor.log_info(info)
        '''
        print(epoch, result)
        print('===============================================')
        '''
        # self._check_best(epoch, score)
        # self.logger(info)
        val_score = {f"val_{k}": val_score[k] for k in val_score.keys()}
        test_score = {f"test_{k}": test_score[k] for k in test_score.keys()}
        score = val_score
        score.update(test_score)
        return score
