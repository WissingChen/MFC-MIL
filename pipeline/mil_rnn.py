from .base import BasePipeline
import numpy as np
import torch
from torch.nn import functional as F
import time
import torch


class MILRNNPipeline(BasePipeline):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        checkpoint = torch.load(f"output/avg_mil_{self.cfgs['dataset']['source']}/checkpoint/model_best.pth")
        # print(checkpoint.keys())
        self.model.encoder.load_state_dict(checkpoint['state_dict'])

        # MIL is frozen, so we only use mil model onece
        self.bag_dataloader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_dataloader)
        self.train_dataloader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_dataloader.dataset.set_mode('selected_bag')

        self.val_dataloader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.val_dataloader)
        self.val_dataloader.dataset.top_k_select(pred, is_in_bag=True, inference=True)
        self.val_dataloader.dataset.set_mode('selected_bag')

        self.test_dataloader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.test_dataloader)
        self.test_dataloader.dataset.top_k_select(pred, is_in_bag=True, inference=True)
        self.test_dataloader.dataset.set_mode('selected_bag')

    def _train_iter(self, epoch):
        loss = self._get_loss()
        self.monitor.log_info(f'\n[train] Epoch: [{epoch}/{self.cfgs["optim"]["epochs"]}]\tLoss: {loss}\n')
    
    def _get_loss(self):

        self.model.encoder.eval()
        self.model.classifier.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_dataloader):
            # [B, k, N] -> [B, K, N]
            inputs = feature.cuda()
            target = target.cuda()
            B = inputs.size(0)
            output = self.model(inputs)

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
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                inputs = feature.cuda()
                output = self.model.encoder(inputs.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                print(f'\r[train] inference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["model"]["num_classes"]])
        return probs

    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                inputs = feature.cuda()
                output = self.model.inference(inputs)
                probs.append(output.detach().cpu().numpy())
                print(f'\r[test] inference progress: {i+1}/{len(loader)}', end='')
        return probs


    def _train_epoch(self, epoch):
        # loop throuh epochs
        self._train_iter(epoch)
        # Validation
        val_pred = self.inference(self.val_dataloader)
        val_score = self.metric(self.val_dataloader.dataset.targets, val_pred)
        test_pred = self.inference(self.test_dataloader)
        test_score = self.metric(self.test_dataloader.dataset.targets, test_pred)

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

