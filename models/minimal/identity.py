import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)
