"""Get DataLoader by dataset name. kwargs are passed to the torchvision dataset."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.sampling import get_stratified_subset


def get_dataset(
    name: str,
    split: str = "test",
    fraction: float = 1.0,
    batch_size: int = 64,
    seed: int = 42,
    data_dir: str = "data",
    **kwargs,
):
    """
    Return (dataset, loader). dataset is the full underlying dataset (for .classes);
    loader iterates over the full set or a stratified subset when fraction < 1.
    kwargs are passed through to the torchvision dataset.
    """
    is_train = split == "train"

    if name == "mnist":
        full = datasets.MNIST(
            root=data_dir, 
            train=is_train, 
            download=True, 
            transform=transforms.ToTensor(), 
            **kwargs
            )
    elif name == "fashion_mnist":
        full = datasets.FashionMNIST(
            root=data_dir, 
            train=is_train, 
            download=True, 
            transform=transforms.ToTensor(), 
            **kwargs
            )
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'mnist' or 'fashion_mnist'.")

    if fraction >= 1.0:
        to_load = full
    else:
        to_load = get_stratified_subset(full, fraction=fraction, seed=seed)

    loader = DataLoader(to_load, batch_size=batch_size, shuffle=False, num_workers=0)
    return full, loader
