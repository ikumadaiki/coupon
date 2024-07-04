import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        lr: float,
    ):
        self.num_epochs = num_epochs
        self.lr = lr

    def train(self, train_dl: DataLoader, model: nn.Module) -> nn.Module:  # type: ignore
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            for batch in train_dl:
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"].backward()
                optimizer.step()

        return model

    def predict(self, dl: DataLoader, model: nn.Module) -> torch.Tensor:  # type: ignore
        model.eval()
        predictions = []
        for batch in dl:
            output = model(**batch)
            predictions.append(output["pred"])
        predictions = torch.cat(predictions, dim=0)

        return predictions
    
    def save_model(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)
