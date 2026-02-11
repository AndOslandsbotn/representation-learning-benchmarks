import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
from .base_embedding import BaseEmbedding

class LaplacianEigenmapsEmbedding(BaseEmbedding):
    def __init__(self, n_components=3, n_neighbors=10):
        super().__init__(n_components)
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        W = kneighbors_graph(X, self.n_neighbors, mode='connectivity')
        W = 0.5 * (W + W.T)

        D = np.diag(W.sum(axis=1).A1)
        L = D - W.toarray()

        _, vecs = eigsh(L, k=self.n_components + 1, sigma=0.0, which='LM')
        return vecs[:, 1:self.n_components+1]
