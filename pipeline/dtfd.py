from .transmil import TransMILPipeline
from utils import build_optimizer
import numpy as np
import torch
from torch.nn import functional as F
import random
import torch
from utils import build_lr_scheduler


class DTFDPipeline(TransMILPipeline):
    def __init__(self, cfgs):
        super(DTFDPipeline, self).__init__(cfgs)
        trainable_parameters_A = []
        trainable_parameters_A += list(self.model.classifier.parameters())
        trainable_parameters_A += list(self.model.attention.parameters())
        trainable_parameters_A += list(self.model.dimReduction.parameters())

        trainable_parameters_B = list(self.model.attCls.parameters())

        trainable_parameters_A += list(self.model.mfc.parameters())

        self.optimizer = build_optimizer(cfgs, trainable_parameters_A)
        self.optimizer1 = build_optimizer(cfgs, trainable_parameters_B)

        self.lr_scheduler = build_lr_scheduler(cfgs, self.optimizer, len(self.train_dataloader))
        self.lr_scheduler1 = build_lr_scheduler(cfgs, self.optimizer1, len(self.train_dataloader))

    def _train_iter(self, epoch):
        loss = self._get_loss()

        self.monitor.log_info(f'\n[train] Epoch: [{epoch}/{self.cfgs["optim"]["epochs"]}]\tLoss: {loss}\n')
    
    def _get_loss(self):
        self.model.train()
        running_loss = 0.
        for i, (feature, _target, slide_id) in enumerate(self.train_dataloader):
            # [1*k, N] -> [B*K, N]
            feature = feature.cuda().float()
            # target = torch.tensor([0., 1.]).cuda() if _target == 1 else torch.tensor([1., 0.]).cuda()
            target = _target.cuda()

            out = self.model(feature, target)
            # Calculate and backpropagate loss for the first tier
            loss_A = self.criterion(out['slide_sub_preds'], out['slide_sub_labels'])
            self.optimizer.zero_grad()
            loss_A.backward(retain_graph=True)
            running_loss += loss_A.item()

            # Second tier optimization
            loss_B = self.criterion(out['gSlidePred'], out['label']).mean()
            self.optimizer1.zero_grad()
            loss_B.backward()
            running_loss += loss_B.item()

            # Clip gradients and update weights
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.optimizer1.step()
            print(f"\r[train] backward progress: {(i+1)/len(self.train_dataloader):.1%}", end='')

            # Step schedulers
            self.lr_scheduler.step()
            self.lr_scheduler1.step()
        return running_loss/len(self.train_dataloader)

    def inference(self, loader, k=5):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                feature = feature.cuda().float()
                output = self.model(feature, target)['gSlidePred'].softmax(dim=1)
                probs.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                print(f'\r[test] inference progress: {i+1}/{len(loader)}', end='')
        return probs, targets
