import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.multi_level import DownSample, MultiLevelFuse
from modules.causal import InterventionOnline, InstanceMemory
from modules.fre_domain import FeatureFreProcessing

class Attention(nn.Module):
    def __init__(self, cfgs):
        super(Attention, self).__init__()
        self.input_dim = cfgs['model']["input_dim"]
        self.embed_dim = cfgs['model']["embed_dim"]

        self.L = self.embed_dim
        self.D = self.embed_dim
        self.K = 1

        self.aggregator = nn.Linear(self.input_dim, self.embed_dim)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            # nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            # nn.Sigmoid()
        )

        ########
        self.down = DownSample(self.embed_dim)
        self.fuse = MultiLevelFuse()

        self.FFP = FeatureFreProcessing(self.embed_dim)
        self.FFP_ll = FeatureFreProcessing(self.embed_dim)

        self.memory = InstanceMemory(d_model=self.embed_dim)
        self.memory_ll = InstanceMemory(32, self.embed_dim)
        self.causal = InterventionOnline(self.embed_dim, self.embed_dim)
        self.causal_ll = InterventionOnline(self.embed_dim, self.embed_dim)
    
    def modify(self, feat, cls_token):
        feat = feat.reshape([1, -1, 512])
        cls_token = cls_token.reshape([1, 1, 512])

        feat_hl, feat_ll = self.down(feat)
        feat_hl = self.FFP(feat_hl)
        feat_ll = self.FFP_ll(feat_ll)

        feat_hat_hl = self.memory(feat_hl)
        feat_hat_ll = self.memory_ll(feat_ll)

        cls_token = self.causal(cls_token, feat_hat_hl) + self.causal(cls_token, feat_hat_ll) + cls_token
        cls_token = cls_token.reshape(1, 512)
        return cls_token

    def forward(self, x):
        feat = self.aggregator(x)  # BxLxN

        attn = self.attention(feat).softmax(dim=1) # BxLx1
        cls_token = torch.matmul(attn.transpose(-1, -2), feat).squeeze(1)  # Bx1xN

        ## MFC-MIL
        cls_token = self.modify(feat, cls_token)

        logits = self.classifier(cls_token)

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part = nn.Sequential(
            nn.Linear(512, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Sigmoid()
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.feature_extractor_part(x)  # NxL

        attn_V = self.attention_V(x)  # NxD
        attn_U = self.attention_U(x)  # NxD
        attn = self.attention_weights(attn_V * attn_U) # element wise multiplication # NxK
        attn = torch.transpose(attn, 1, 0)  # KxN
        attn = F.softmax(attn, dim=1)  # softmax over N

        m = torch.mm(attn, x)  # KxL

        y = self.classifier(m)

        return y, attn

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y
