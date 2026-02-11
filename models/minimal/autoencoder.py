import torch
import torchvision

if __name__ == '__main__':
  fashion_minist = torchvision.datasets.FashionMNIST(
      root='data', train=True, download=True,
      transform=torchvision.transforms.ToTensor()
  )
  d = 3