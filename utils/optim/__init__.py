from .loss import compute_ce_loss
from .lr_scheduler import build_lr_scheduler
from .optimizer import build_optimizer
from torch import nn


loss_fns = {"ce": nn.CrossEntropyLoss, 'nll': nn.NLLLoss, 'bce': nn.BCEWithLogitsLoss}
