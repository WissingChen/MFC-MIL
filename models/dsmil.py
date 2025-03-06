import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.multi_level import DownSample, MultiLevelFuse
from modules.causal import InterventionOnline, InstanceMemory
from modules.fre_domain import FeatureFreProcessing

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=512, n_classes=2):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_size, n_classes)
    def forward(self, x):
        feats = x
        classes = self.fc(feats)
        return feats, classes

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class=2, dropout_v=0.0, nonlinear=True, passing_v=False, confounder_path=False): # K, L, N
        super(BClassifier, self).__init__()
        self.input_size = input_size
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

        ########
        self.down = DownSample(self.input_size)
        self.fuse = MultiLevelFuse()

        self.FFP = FeatureFreProcessing(self.input_size)
        self.FFP_ll = FeatureFreProcessing(self.input_size)

        self.memory = InstanceMemory(d_model=self.input_size)
        self.memory_ll = InstanceMemory(32,d_model=self.input_size)
        self.causal = InterventionOnline(input_size, 512)
        self.causal_ll = InterventionOnline(input_size, 512)
        

    def modify(self, feat, cls_token):
        feat = feat.reshape([1, -1, self.input_size])
        cls_token = cls_token.reshape([1, 2, self.input_size])
        feat_hl, feat_ll = self.down(feat)

        feat_hl = self.FFP(feat_hl)
        feat_ll = self.FFP_ll(feat_ll)

        feat_hat_hl = self.memory(feat_hl)
        feat_hat_ll = self.memory_ll(feat_ll)

        cls_token = self.causal(cls_token, feat_hat_hl) + self.causal_ll(cls_token, feat_hat_ll) + cls_token
        cls_token = cls_token.reshape(1, 2, self.input_size)
        return cls_token
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V

        B = self.modify(V, B)

        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B


class DSMIL(nn.Module):
    def __init__(self, cfgs):
        super(DSMIL, self).__init__()
        self.input_dim = cfgs['model']["input_dim"]
        self.embed_dim = cfgs['model']["embed_dim"]
        self.i_classifier = FCLayer(self.input_dim)
        self.b_classifier = BClassifier(input_size=self.input_dim, output_class=2)

    def forward(self, x):
        x = x.reshape([-1, self.input_dim])
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        logits = prediction_bag
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, "i_logits": classes, "bag_feature": B}
        return results_dict
        # return classes, prediction_bag, A, B

