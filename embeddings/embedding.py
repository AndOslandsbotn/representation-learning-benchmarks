"""Embedding reduction (t-SNE, Laplacian) and pipeline: extract → reduce → save."""
import torch
from joblib import Memory
from sklearn.manifold import TSNE, SpectralEmbedding

from embeddings.extractor import extract_activations
from embeddings.visualization import EmbeddingVisualizer

memory = Memory(location="cache/embeddings/")


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
    model_config,
    embedding_configs,
    dataset_name="data",
    results_dir="experiments/results",
    device=None,
):
    """
    For one model: extract activations at each layer, then for each embedding_config
    reduce to 3D and save a plot.

    model_config: dict with
        - name: str (e.g. "identity", "cnn_proto1")
        - model_factory: callable() -> nn.Module
        - layer_specs: dict mapping layer_name -> callable(model) -> layer
        - weights: optional str path to state_dict
    embedding_configs: list of dicts, each with "method" and method-specific kwargs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_config["name"]
    model = model_config["model_factory"]()
    if model_config.get("weights"):
        model.load_state_dict(torch.load(model_config["weights"], map_location="cpu"))
    model.eval()

    for layer_name, get_layer in model_config["layer_specs"].items():
        layer = get_layer(model)
        X, y = extract_activations(model, layer, loader, device=device)
        y_names = [str(dataset.classes[i]) for i in y]

        for emb_cfg in embedding_configs:
            method = emb_cfg["method"]
            kwargs = {k: v for k, v in emb_cfg.items() if k != "method"}
            X_emb = compute_embedding(X, method=method, **kwargs)
            title = f"{dataset_name} {name} {layer_name} + {method.upper()}"
            save_path = f"{results_dir}/{dataset_name}_{name}_{layer_name}_{method}.html"
            EmbeddingVisualizer(title=title).plot(X_emb, y_names, save_path=save_path)
