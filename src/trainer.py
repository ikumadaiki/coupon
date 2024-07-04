import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        lr: float,
    ):
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.lr = lr

    def train(self, dl: DataLoader, model: nn.Module) -> nn.Module:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            for batch in dl:
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"].backward()
                optimizer.step()

        return model
