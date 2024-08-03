from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from src.early_stopping import EarlyStopping
from src.model.common import make_loader

# NNのランダム性を固定
torch.manual_seed(42)


class Trainer:
    def __init__(self, num_epochs: int, patience: int = 10) -> None:
        self.num_epochs = num_epochs
        self.patience = patience

    def train(
        self,
        train_dl: DataLoader,  # type: ignore
        val_dl: DataLoader,  # type: ignore
        model: nn.Module,
        lr: float,
        weight_decay: float,
        method: str,
    ) -> Tuple[nn.Module, float, List[float], List[float]]:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_epochs * len(train_dl),
        )
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        train_loss_history: List[float] = []
        val_loss_history: List[float] = []
        loss = np.inf
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            model.train()
            total_train_loss: float = 0.0
            count_batch: int = 0
            average_loss: float = 0.0
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
                average_loss = total_train_loss / count_batch
            train_loss_history.append(total_train_loss / count_batch)

            # モデルのパラメータをprint
            # if epoch % 10 == 0:
            #     print(f"Parameters after epoch {epoch}:")
            #     for name, param in model.named_parameters():
            #         print(f"{name} - {param.size()}")
            #         print(param.data)
            #     import pdb; pdb.set_trace()

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

            # EarlyStoppingの呼び出し
            early_stopping(total_val_loss / count_val_batch, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return (
            model,
            total_val_loss / count_val_batch,
            train_loss_history,
            val_loss_history,
        )

    def grid_search(
        self,
        train_dataset: dict[str, NDArray[Any]],
        val_dataset: dict[str, NDArray[Any]],
        model: nn.Module,
        lr_list: List[float],
        batch_size_list: List[int],
        weight_decay_list: List[float],
        method: str,
        model_name: str,
    ) -> Tuple[nn.Module, float, float, int, float]:
        best_model = None
        best_val_loss = np.inf
        best_lr = None
        best_batch_size = None

        for lr in lr_list:
            for batch_size in batch_size_list:
                for weight_decay in weight_decay_list:
                    train_dl = make_loader(
                        dataset=train_dataset,
                        model_name=model_name,
                        batch_size=batch_size,
                        train_flg=True,
                        method=method,
                    )
                    val_dl = make_loader(
                        dataset=val_dataset,
                        model_name=model_name,
                        batch_size=batch_size,
                        train_flg=True,
                        method=method,
                    )
                    model, val_loss, train_loss_history, val_loss_history = self.train(
                        train_dl=train_dl,
                        val_dl=val_dl,
                        model=model,
                        lr=lr,
                        weight_decay=weight_decay,
                        method=method,
                    )
                    if val_loss < best_val_loss:
                        best_model = model
                        best_val_loss = val_loss
                        best_lr = lr
                        best_batch_size = batch_size
                        best_train_loss_history, best_val_loss_history = (
                            train_loss_history,
                            val_loss_history,
                        )
        self.plot_loss(best_train_loss_history, best_val_loss_history, method)

        return (
            best_model,
            best_val_loss,
            best_lr,
            best_batch_size,
            weight_decay,
        )  # type: ignore

    def plot_loss(
        self, train_loss: List[float], val_loss: List[float], method: str
    ) -> None:
        plt.clf()
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_{method}.png")

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
