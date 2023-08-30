from random import sample

from typing import Tuple
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import numpy as np
import os


def create_datasets(root_folder: str) -> Tuple[CelebA, CelebA]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )
    train_data = CelebA(
        root=root_folder, split="train", transform=transform, download=True
    )
    test_data = CelebA(
        root=root_folder, split="test", transform=transform, download=True
    )
    return train_data, test_data


def create_dataloaders(
    train_data: CelebA,
    test_data: CelebA,
    batch_size: int,
    num_train_samples: int | None,
) -> Tuple[DataLoader, DataLoader]:
    if num_train_samples is not None:
        subset = Subset(
            train_data,
            sample(range(len(train_data)), k=num_train_samples),
        )
        data = subset
    else:
        data = train_data
    num_cpus = os.cpu_count()
    num_workers = min(num_cpus // 2, 2) if num_cpus is not None else 0
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return train_loader, test_loader
