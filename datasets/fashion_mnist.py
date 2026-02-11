from typing import List
from torchvision import datasets, transforms

class FashionMNISTDataset:
    def __init__(self, root="data", train=False):
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

        self.name = "fashion_mnist"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_classes(self):
        return self.dataset.classes
    
    def map_labels(self, labels) -> List[str]:
        return [self.dataset.classes[l] for l in labels]