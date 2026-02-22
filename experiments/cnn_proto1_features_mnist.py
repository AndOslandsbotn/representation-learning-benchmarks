"""Extract activations from selected CNN proto1 layers on MNIST and embed."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.minimal.cnn import CNNPrototype1
from embeddings.extractor import extract_activations
from embeddings.embedding import compute_embedding
from embeddings.visualization import EmbeddingVisualizer

WEIGHTS_PATH = "training/runs/cnn_proto1/weights.pt"
DATA_DIR = "data"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "experiments/results"

LAYER_SPECS = [
    ("start_block", lambda m: m.conv[2]),
    ("end_layer", lambda m: m.conv),
]

EMBEDDING_CONFIGS = [
    {"method": "tsne", "n_components": 3, "perplexity": 30},
    {"method": "laplacian", "n_components": 3, "n_neighbors": 10},
]


def main():
    model = CNNPrototype1(num_classes=10)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    test_ds = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for layer_name, get_layer in LAYER_SPECS:
        layer = get_layer(model)
        X, y = extract_activations(model, layer, loader, device=DEVICE)
        y_names = [str(test_ds.classes[i]) for i in y]
        print(f"Layer {layer_name}: activations shape {X.shape}, labels shape {y.shape}")

        for cfg in EMBEDDING_CONFIGS:
            method = cfg["method"]
            kwargs = {k: v for k, v in cfg.items() if k != "method"}
            X_emb = compute_embedding(X, method=method, **kwargs)
            title = f"CNN proto1 MNIST {layer_name} + {method.upper()}"
            save_path = f"{RESULTS_DIR}/cnn_proto1_mnist_{layer_name}_{method}.html"
            EmbeddingVisualizer(title=title).plot(X_emb, y_names, save_path=save_path)


if __name__ == "__main__":
    main()
