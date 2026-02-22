"""Run identity model on all datasets: extract → embed → save."""
import torch
from models.minimal.identity import IdentityModel
from utils.datasets import get_dataset
from fsinspector import embedding

RESULTS_DIR = "experiments/results"
MODEL_NAME = "identity"
DATASETS = ["mnist", "fashion_mnist"]

LAYER_SPECS = {"identity": lambda m: m.flatten}

EMBEDDING_CONFIGS = [
    {"method": "tsne", "n_components": 3, "perplexity": 30},
    {"method": "laplacian", "n_components": 3, "n_neighbors": 10},
]


def main():
    model = IdentityModel()
    model.eval()

    for dataset_name in DATASETS:
        dataset, loader = get_dataset(dataset_name, fraction=1.0, batch_size=64)

        for layer_name, get_layer in LAYER_SPECS.items():
            layer = get_layer(model)
            for emb_cfg in EMBEDDING_CONFIGS:
                method = emb_cfg["method"]
                kwargs = {k: v for k, v in emb_cfg.items() if k != "method"}
                embedding.run(
                    dataset,
                    loader,
                    model,
                    layer,
                    layer_name,
                    MODEL_NAME,
                    method,
                    kwargs,
                    dataset_name=dataset_name,
                    results_dir=RESULTS_DIR,
                )


if __name__ == "__main__":
    main()
