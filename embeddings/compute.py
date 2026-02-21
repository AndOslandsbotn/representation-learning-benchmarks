from joblib import Memory
from sklearn.manifold import TSNE, SpectralEmbedding

memory = Memory(location="cache/embeddings/")

@memory.cache
def compute_embedding(X, method: str = 'tsne', **kwargs):
    if method == 'tsne':
        model = TSNE(**kwargs)
        return model.fit_transform(X)
    if method == 'laplacian':
        model = SpectralEmbedding(**kwargs)
        return model.fit_transform(X)
    raise ValueError(f"Unknown embedding method: {method}")
