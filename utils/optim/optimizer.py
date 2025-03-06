import torch
import math
from utils.optim.lookahead import Lookahead, create_lookahead_optimizer


def build_optimizer(cfgs, model):
    optim = cfgs["optim"]["optimizer"]
    weight_decay = cfgs["optim"]["weight_decay"]
    lr = cfgs["optim"]["lr"]
    amsgrad = cfgs["optim"]["amsgrad"]
    params_for_optimization = model if isinstance(model, list) else list(p for p in model.parameters() if p.requires_grad)
    if optim == 'SGD' or optim == "Adamax":
        optimizer = getattr(torch.optim, optim)(
            [{'params': params_for_optimization, 'lr': lr}],
            weight_decay=weight_decay,
        )
    elif optim == 'lookahead':
        optimizer = create_lookahead_optimizer(
            {"opt": "lookahead_radam",
            'lr': lr,
            "opt_eps": None, 
            "opt_betas": None,
            "momentum": None, 
            "weight_decay": weight_decay},
            model
            )
    elif optim =='Adam':
        optimizer = getattr(torch.optim, optim)(
            [{'params': params_for_optimization, 'lr': lr}],
            weight_decay=weight_decay,
            betas= cfgs['optim']['betas'] if cfgs['optim']['betas'] else [0.95, 0.99],
            amsgrad=amsgrad,
        )
    else:
        optimizer = getattr(torch.optim, optim)(
            [{'params': params_for_optimization, 'lr': lr}],
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    return optimizer

