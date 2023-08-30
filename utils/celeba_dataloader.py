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
            indices=np.random.permutation(len(train_data))[:num_train_samples],
        )
        data = subset
    else:
        data = train_data
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=min(os.cpu_count() // 2, 2),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=min(os.cpu_count() // 2, 2),
        persistent_workers=True,
    )
    return train_loader, test_loader
