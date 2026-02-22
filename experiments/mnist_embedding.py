"""MNIST: run identity and cnn_proto1 via embeddings.embedding.run()."""
from models.minimal.identity import IdentityModel
from models.minimal.cnn import CNNPrototype1
from utils.datasets import get_dataset
from embeddings import embedding

RESULTS_DIR = "experiments/results"
CNN_WEIGHTS = "training/runs/cnn_proto1/mnist/weights.pt"

MODEL_CONFIGS = [
    {
        "name": "identity",
        "model_factory": lambda: IdentityModel(),
        "layer_specs": {"identity": lambda m: m.flatten},
    },
    {
        "name": "cnn_proto1",
        "model_factory": lambda: CNNPrototype1(num_classes=10),
        "weights": CNN_WEIGHTS,
        "layer_specs": {
            "start_block": lambda m: m.conv[2],
            "end_layer": lambda m: m.conv,
        },
    },
]

EMBEDDING_CONFIGS = [
    {"method": "tsne", "n_components": 3, "perplexity": 30},
    {"method": "laplacian", "n_components": 3, "n_neighbors": 10},
]


def main():
    dataset, loader = get_dataset("mnist", fraction=1.0, batch_size=64)

    for model_cfg in MODEL_CONFIGS:
        embedding.run(
            dataset,
            loader,
            model_cfg,
            EMBEDDING_CONFIGS,
            dataset_name="mnist",
            results_dir=RESULTS_DIR,
        )


if __name__ == "__main__":
    main()
