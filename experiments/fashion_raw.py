from datasets.fashion_mnist import FashionMNISTDataset
from models.minimal.identity import IdentityModel
from analysis.extractor import RepresentationExtractor
from embeddings.tsne import TSNEEmbedding
from utils.visualization import PlotlyVisualizer


def main():
    dataset = FashionMNISTDataset(train=False)
    model = IdentityModel()

    extractor = RepresentationExtractor(
        model=model,
        dataset=dataset,
        level=-1,
        batch_size=256,
    )

    X, y = extractor.extract()

    embedding = TSNEEmbedding(n_components=3)
    X_embedded = embedding.fit_transform(X)

    visualizer = PlotlyVisualizer(
        title="Fashion-MNIST Identity + t-SNE"
    )

    y_strings = dataset.map_labels(y)
    visualizer.plot(
        X_embedded,
        y_strings,
        save_path="identity_tsne_fashion.html"
    )


if __name__ == "__main__":
    main()
