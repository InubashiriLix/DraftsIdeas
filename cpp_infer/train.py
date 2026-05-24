from numpy import test
from model import ModelCnn
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

LR: float = 0.001
NUM_EPOCHS: int = 5
EXPORT: bool = True
EXPORT_NAME = "./model.pt"

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    model = ModelCnn(28, 28, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    if EXPORT:
        torch.save(model.state_dict(), EXPORT_NAME)
        print(f"Model saved to {EXPORT_NAME}")


if __name__ == "__main__":
    main()
