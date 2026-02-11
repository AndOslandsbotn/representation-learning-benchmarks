from sklearn.manifold import TSNE
from .base_embedding import BaseEmbedding

class TSNEEmbedding(BaseEmbedding):
    def __init__(self, n_components=3, perplexity=30, random_state=42):
        super().__init__(n_components)
        self.model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )

    def fit_transform(self, X):
        return self.model.fit_transform(X)
