"""Run cnn_proto1 on all datasets: load weights per dataset, extract layers → embed → save."""
import torch
from models.minimal.cnn import CNNPrototype1
from utils.datasets import get_dataset
from fsinspector import embedding

RESULTS_DIR = "experiments/results"
MODEL_NAME = "cnn_proto1"
DATASETS = ["mnist", "fashion_mnist"]
WEIGHTS_DIR = "training/runs/cnn_proto1"

LAYER_SPECS = {
    "start_block": lambda m: m.conv[2],
    "end_layer": lambda m: m.conv,
}

EMBEDDING_CONFIGS = [
    {"method": "tsne", "n_components": 3, "perplexity": 30},
    {"method": "laplacian", "n_components": 3, "n_neighbors": 10},
]


def main():
    for dataset_name in DATASETS:
        dataset, loader = get_dataset(dataset_name, fraction=1.0, batch_size=64)

        model = CNNPrototype1(num_classes=10)
        model.load_state_dict(
            torch.load(f"{WEIGHTS_DIR}/{dataset_name}/weights.pt", map_location="cpu")
        )
        model.eval()

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
