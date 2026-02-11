from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_representation(self, x: torch.Tensor, level: int = -1) -> torch.Tensor:
        """
        Returns a 2D tensor of shape (batch_size, feature_dim) 
        """
        pass

    @abstractmethod
    def available_levels(self) -> List[int]:
        """
        Returns list of valid representation levels.
        """
        pass

