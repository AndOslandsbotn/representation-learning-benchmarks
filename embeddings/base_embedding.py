from abc import abstractmethod
import numpy as np

class BaseEmbedding():
    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass
