from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from src.early_stopping import EarlyStopping

# NNのランダム性を固定
torch.manual_seed(42)


class Trainer:
    def __init__(
        self, num_epochs: int, weight_decay: float = 0.0, patience: int = 10
    ) -> None:
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.patience = patience

    def train(
        self,
        train_dl: DataLoader,  # type: ignore
        val_dl: DataLoader,  # type: ignore
        model: nn.Module,
        lr: float,
        method: str,
    ) -> Tuple[nn.Module, float]:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)

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

        return model, total_val_loss / count_val_batch

    def grid_search(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        model: nn.Module,
        lr_list: List[float],
        batch_size_list: List[int],
        method: str,
    ) -> Tuple[nn.Module, float, float, int]:
        best_model = None
        best_val_loss = np.inf
        best_lr = None
        best_batch_size = None

        for lr in lr_list:
            for batch_size in batch_size_list:
                print(f"lr: {lr}, batch_size: {batch_size}")
                model, val_loss = self.train(
                    train_dl=train_dl, val_dl=val_dl, model=model, lr=lr, method=method
                )
                if val_loss < best_val_loss:
                    best_model = model
                    best_val_loss = val_loss
                    best_lr = lr
                    best_batch_size = batch_size
        return best_model, best_val_loss, best_lr, best_batch_size

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
