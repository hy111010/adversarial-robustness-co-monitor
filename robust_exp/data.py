from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def _limit(indices: list[int], limit: int | None) -> list[int]:
    return indices if limit is None else indices[: min(limit, len(indices))]


def make_loaders(
    root: str,
    batch_size: int,
    workers: int,
    seed: int,
    val_size: int = 5000,
    limit_train: int | None = None,
    limit_val: int | None = None,
    dataset: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    import torch

    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    eval_transform = transforms.ToTensor()
    dataset_class = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}.get(dataset)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    train_set_aug = dataset_class(root=root, train=True, download=True, transform=train_transform)
    train_set_eval = dataset_class(root=root, train=True, download=False, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_set_aug), generator=generator).tolist()
    val_indices = _limit(order[:val_size], limit_val)
    train_indices = _limit(order[val_size:], limit_train)

    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        Subset(train_set_aug, train_indices), shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(Subset(train_set_eval, val_indices), shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def make_test_loader(
    root: str, batch_size: int, workers: int, dataset: str = "cifar10"
) -> DataLoader:
    import torch

    dataset_class = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}.get(dataset)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    test_dataset = dataset_class(root=root, train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
