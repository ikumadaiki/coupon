from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

# NNのランダム性を固定
torch.manual_seed(42)


class Trainer:
    def __init__(self, num_epochs: int, lr: float):
        self.num_epochs = num_epochs
        self.lr = lr

    def train(
        self,
        train_dl: DataLoader,  # type: ignore
        val_dl: DataLoader,  # type: ignore
        model: nn.Module,  # type: ignore
        method: str,
    ) -> nn.Module:  # type: ignore
        optimizer = AdamW(model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_epochs * len(train_dl),
        )
        train_loss_history: List[float] = []
        val_loss_history: List[float] = []
        loss = np.inf
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            model.train()
            total_train_loss: float = 0.0
            count_batch: int = 0
            average_loss = 0
            total = len(train_dl)
            desc = f"Epoch {epoch} AVG Loss: {average_loss:.4f}"
            for batch in tqdm(train_dl, desc=desc, leave=False):
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"].backward()
                optimizer.step()

                loss = output["loss"].item()
                total_train_loss += loss
                count_batch += 1

            scheduler.step()
            train_loss_history.append(total_train_loss / count_batch)
            model.eval()  # モデルを評価モードに設定
            total_val_loss = 0.0
            count_val_batch = 0
            with torch.no_grad():
                for val_batch in tqdm(val_dl, total=total, desc=desc, leave=False):
                    val_output = model(**val_batch)
                    val_loss = val_output["loss"].item()
                    total_val_loss += val_loss
                    count_val_batch += 1
            val_loss_history.append(total_val_loss / count_val_batch)

        plt.clf()
        plt.plot(train_loss_history, label="Train")
        plt.plot(val_loss_history, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"train_val_loss_{method}.png")
        return model

    def predict(self, dl: DataLoader, model: nn.Module) -> NDArray[Any]:  # type: ignore
        model.eval()
        with torch.no_grad():
            predictions = []
            for batch in dl:
                output = model(**batch)
                pred: torch.Tensor = output["pred"]
                predictions.append(pred.detach().cpu().numpy())

            return np.concatenate(predictions, axis=0)

    def save_model(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)

    def save_predictions(self, predictions: NDArray[Any], path: str) -> None:
        np.save(path, predictions)
