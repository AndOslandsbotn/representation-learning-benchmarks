from torch.utils.data import Subset
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


def get_subset(dataset, n_samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    return Subset(dataset, indices)

def get_stratified_subset(
    dataset,
    samples_per_class: int | None = None,
    fraction: float | None = None,
    seed: int = 0,
):

    if samples_per_class is None and fraction is None:
        raise ValueError("Provide either samples_per_class or fraction")
    targets = np.array(dataset.targets)
    n = len(dataset)
    rng = np.random.default_rng(seed)

    if fraction is not None:
        if fraction >= 1.0 or fraction * n >= n:
            idx = np.arange(n)
            rng.shuffle(idx)
            return Subset(dataset, idx)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
        idx, _ = next(sss.split(np.arange(n), targets))
        return Subset(dataset, idx)

    n_classes = len(np.unique(targets))
    n_total = n_classes * samples_per_class
    if n_total >= n:
        idx = np.arange(n)
        rng.shuffle(idx)
        return Subset(dataset, idx)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_total, random_state=seed)
    idx, _ = next(sss.split(np.arange(n), targets))
    return Subset(dataset, idx)


def split_train_val(dataset, val_fraction: float = 0.1, seed: int = 0):
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(indices, targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
