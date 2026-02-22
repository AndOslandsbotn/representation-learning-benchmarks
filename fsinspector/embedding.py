"""Embedding reduction (t-SNE, Laplacian) and pipeline: extract → reduce → save."""
import torch
from joblib import Memory
from sklearn.manifold import TSNE, SpectralEmbedding

from fsinspector.extractor import extract_activations
from fsinspector.visualization import EmbeddingVisualizer

memory = Memory(location="cache/fsinspector/")


@memory.cache
def compute_embedding(X, method: str = "tsne", **kwargs):
    if method == "tsne":
        model = TSNE(**kwargs)
        return model.fit_transform(X)
    if method == "laplacian":
        model = SpectralEmbedding(**kwargs)
        return model.fit_transform(X)
    raise ValueError(f"Unknown embedding method: {method}")


def run(
    dataset,
    loader,
    model,
    layer,
    layer_name: str,
    model_name: str,
    embedding_method: str,
    embedding_kwargs: dict,
    dataset_name: str = "data",
    results_dir: str = "experiments/results",
    device=None,
):
    """
    One model, one layer, one embedding method: extract activations, reduce to 3D, save plot.
    Looping over models, layers, and embedding methods is done in the experiment script.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = extract_activations(model, layer, loader, device=device)
    y_names = [str(dataset.classes[i]) for i in y]
    X_emb = compute_embedding(X, method=embedding_method, **embedding_kwargs)
    title = f"{dataset_name} {model_name} {layer_name} + {embedding_method.upper()}"
    save_path = f"{results_dir}/{dataset_name}_{model_name}_{layer_name}_{embedding_method}.html"
    EmbeddingVisualizer(title=title).plot(X_emb, y_names, save_path=save_path)
