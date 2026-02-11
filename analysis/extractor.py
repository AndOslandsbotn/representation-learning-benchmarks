import torch
import numpy as np


class RepresentationExtractor:

    def __init__(
        self,
        model,
        dataset,
        level="final",
        device="cpu",
        batch_size=64,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.level = level
        self.device = device
        self.batch_size = batch_size

    def extract(self):
        self.model.eval()

        all_reps = []
        all_labels = []

        with torch.no_grad():

            for start in range(0, len(self.dataset), self.batch_size):
                end = min(start + self.batch_size, len(self.dataset))

                batch = [self.dataset[i] for i in range(start, end)]
                xs, ys = zip(*batch)

                xs = torch.stack(xs).to(self.device)

                reps = self.model.get_representation(
                    xs, level=self.level
                )

                if reps.ndim != 2:
                    raise ValueError(
                        f"Expected (batch_size, feature_dim), got {reps.shape}"
                    )

                all_reps.append(reps.cpu())
                all_labels.extend(ys)

        X = torch.cat(all_reps, dim=0).numpy()
        y = np.array(all_labels)

        return X, y
