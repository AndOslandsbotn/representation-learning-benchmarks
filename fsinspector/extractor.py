"""Extract layer activations via forward hooks for embedding."""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
import torch.nn as nn


class ActivationCollector:
    """Hook callback that stores layer outputs in .features. Use as: layer.register_forward_hook(collector)."""

    def __init__(self):
        self.features = []

    def __call__(self, _module, _input, output):
        self.features.append(output.detach().cpu())


def extract_activations(
    model: nn.Module, layer: nn.Module, dataloader: DataLoader, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations at `layer` for all batches in `dataloader`. Returns (X, y) as numpy arrays; X is (N, D) flattened per sample."""
    model = model.to(device).eval()
    collector = ActivationCollector()
    handle = layer.register_forward_hook(collector)
    labels = []
    try:
        with torch.no_grad():
            for x, y in dataloader:
                model(x.to(device))
                labels.append(y)
        X = torch.cat(collector.features, dim=0).flatten(1).numpy()
        y = torch.cat(labels, dim=0).numpy()
        return X, y
    finally:
        handle.remove()
