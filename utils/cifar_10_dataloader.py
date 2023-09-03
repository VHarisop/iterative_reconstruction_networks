from random import sample

from typing import Tuple
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os


def create_datasets(root_folder: str) -> Tuple[CIFAR10, CIFAR10]:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )
    train_data = CIFAR10(
        root=root_folder,
        train=True,
        transform=transform,
        download=True,
    )
    test_data = CIFAR10(
        root=root_folder,
        train=False,
        transform=transform,
        download=True,
    )
    return train_data, test_data


def create_dataloaders(
    train_data: CIFAR10,
    test_data: CIFAR10,
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
