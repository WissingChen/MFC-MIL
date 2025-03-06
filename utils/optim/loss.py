import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def compute_ce_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, target)
    return loss
