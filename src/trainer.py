from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        lambda_scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.90**epoch
        )
        loss = np.inf
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            for batch in tqdm(
                train_dl, desc=f"Epoch {epoch} loss={loss:.3f}", leave=False
            ):
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"].backward()
                optimizer.step()

                loss = output["loss"].item()

            lambda_scheduler.step()
        return model

    def predict(self, dl: DataLoader, model: nn.Module) -> NDArray[Any]:  # type: ignore
        model.eval()
        predictions = []
        for batch in dl:
            output = model(**batch)
            pred: torch.Tensor = output["pred"]
            predictions.append(pred.detach().cpu().numpy())

        return np.concatenate(predictions, axis=0)
    
    def save_model(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)
