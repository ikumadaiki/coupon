from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# NNのランダム性を固定
torch.manual_seed(42)


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        lr: float,
    ):
        self.num_epochs = num_epochs
        self.lr = lr

    def train(
        self, train_dl: DataLoader, val_dl: DataLoader, model: nn.Module
    ) -> nn.Module:  # type: ignore
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        lambda_scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.90**epoch
        )
        loss = np.inf
        train_loss_history: list = []
        val_loss_history: list = []
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            model.train()
            total_train_loss: float = 0.0
            count_batch: int = 0
            for batch in tqdm(
                train_dl, desc=f"Epoch {epoch} loss={loss:.3f}", leave=False
            ):
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"].backward()
                optimizer.step()

                loss = output["loss"].item()
                total_train_loss += loss
                count_batch += 1

            lambda_scheduler.step()
            train_loss_history.append(total_train_loss / count_batch)
            model.eval()  # モデルを評価モードに設定
            total_val_loss = 0.0
            count_val_batch = 0
            with torch.no_grad():
                for val_batch in val_dl:
                    val_output = model(**val_batch)
                    val_loss = val_output["loss"].item()
                    total_val_loss += val_loss
                    count_val_batch += 1
            val_loss_history.append(total_val_loss / count_val_batch)

        plt.plot(train_loss_history)
        plt.savefig("train_loss.png")
        plt.clf()
        plt.plot(val_loss_history)
        plt.savefig("val_loss.png")
        return model

    def predict(self, dl: DataLoader, model: nn.Module) -> NDArray[Any]:  # type: ignore
        model.eval()
        predictions = []
        import pdb

        pdb.set_trace()
        for batch in dl:
            output = model(**batch)
            pred: torch.Tensor = output["pred"]
            predictions.append(pred.detach().cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save_model(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)
