import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os import makedirs
from os.path import dirname
from models.minimal.cnn import CNNPrototype1
from utils.sampling import split_train_val

DATA_DIR = "data"
BATCH_SIZE = 64
VAL_FRACTION = 0.1
EPOCHS = 5
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = "training/runs/cnn_proto1/weights.pt"

def main():
    transform = transforms.ToTensor()
    train_full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    train_ds, val_ds = split_train_val(train_full, val_fraction=VAL_FRACTION, seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNNPrototype1(num_classes=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        n_batches = len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}  train_loss = {total_loss / n_batches:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"  val_acc = {correct / total:.4f}")
    makedirs(dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("Saved cnn_proto1.pt")


if __name__ == "__main__":
    main()
