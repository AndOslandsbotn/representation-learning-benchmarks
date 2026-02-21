import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.minimal.identity import IdentityModel
from analysis.extractor import extract_activations
from embeddings.compute import compute_embedding
from utils.sampling import get_stratified_subset
from utils.visualization import EmbeddingVisualizer

DATA_DIR = "data"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "experiments/results"
EMBEDDING_CONFIGS = [
    {"method": "tsne", "n_components": 3, "perplexity": 30},
    {"method": "laplacian", "n_components": 3, "n_neighbors": 10},
]

def main():
    dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    samples = get_stratified_subset(dataset, fraction=1.0, seed=42)
    loader = DataLoader(samples, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = IdentityModel()
    X, y = extract_activations(model, model.flatten, loader, device=DEVICE)
    y_names = [dataset.classes[i] for i in y]

    for cfg in EMBEDDING_CONFIGS:
        method = cfg["method"]
        kwargs = {k: v for k, v in cfg.items() if k != "method"}
        X_emb = compute_embedding(X, method=method, **kwargs)
        title = f"Fashion-MNIST Identity + {method.upper()}"
        save_path = f"{RESULTS_DIR}/identity_fashion_{method}.html"
        EmbeddingVisualizer(title=title).plot(X_emb, y_names, save_path=save_path)

if __name__ == "__main__":
    main()
