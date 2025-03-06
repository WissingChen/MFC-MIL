import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.multi_level import DownSample, MultiLevelFuse
from modules.causal import InterventionOnline, InstanceMemory
from modules.fre_domain import FeatureFreProcessing

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x
    



class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class MFC(nn.Module):
    def __init__(self):
        super(MFC, self).__init__()
        self.down = DownSample()
        self.fuse = MultiLevelFuse()

        self.FFP = FeatureFreProcessing()
        self.FFP_ll = FeatureFreProcessing()

        self.memory = InstanceMemory()
        self.memory_ll = InstanceMemory(32)
        self.causal = InterventionOnline(512, 512)
        self.causal_ll = InterventionOnline(512, 512)
    
    def forward(self, feat, cls_token):
        feat = feat.reshape([1, -1, 512])
        cls_token = cls_token.reshape([1, -1, 512])
        feat_hl, feat_ll = self.down(feat)

        feat_hl = self.FFP(feat_hl)
        feat_ll = self.FFP_ll(feat_ll)

        feat_hat_hl = self.memory(feat_hl)
        feat_hat_ll = self.memory_ll(feat_ll)

        cls_token = self.causal(cls_token, feat_hat_hl) + self.causal_ll(cls_token, feat_hat_ll) + cls_token
        cls_token = cls_token.reshape(-1, 512)
        return cls_token

    

class DTFD(nn.Module):
    def __init__(self, cfgs):
        super(DTFD, self).__init__()
        self.input_dim = cfgs['model']["input_dim"]
        self.embed_dim = cfgs['model']["embed_dim"]

        self.classifier = Classifier_1fc(self.embed_dim, 2, 0.1)
        self.attention = Attention(self.embed_dim)
        self.dimReduction = DimReduction(self.input_dim, self.embed_dim, numLayer_Res=0)

        self.attCls = Attention_with_Classifier(L=self.embed_dim, num_cls=2, droprate=0.1)
        
        self.mfc = MFC()
    
    def get_cam_1d(self, classifier, features):
        tweight = list(classifier.parameters())[-2]
        cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
        return cam_maps
    
    def forward(self, x, label, distill="MaxS"):
        instance_per_group = 1
        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []

        # Split bag into chunks
        inputs_pseudo_bags = torch.chunk(x.squeeze(0), instance_per_group, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            slide_sub_labels.append(label)
            subFeat_tensor = subFeat_tensor.cuda()

            # Forward pass through models
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  # 1 x 2
            patch_pred_logits = self.get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

            slide_sub_preds.append(tPredict)
        
         # Concatenate tensors
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup
        
        slide_pseudo_feat = self.mfc(tmidFeat, slide_pseudo_feat)
        # Second tier optimization
        gSlidePred = self.attCls(slide_pseudo_feat)
        out = {'slide_sub_preds': slide_sub_preds, 'slide_sub_labels': slide_sub_labels, 'gSlidePred': gSlidePred, "label": label}
        return out