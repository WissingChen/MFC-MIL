import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import BagDataset, InstanceMILDataset


def build_dataloaders(cfgs, transform=None):
    train_dataset = InstanceMILDataset(cfgs, "train", transform)
    val_dataset = BagDataset(cfgs, "val", transform)
    test_dataset = BagDataset(cfgs, "test", transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfgs["dataset"]["num_workers"],
        shuffle=False,
    )


    return train_loader, val_loader, test_loader