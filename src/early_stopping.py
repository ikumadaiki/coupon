import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience: int = 7, verbose: bool = False, delta: float = 0
    ) -> None:
        """
        Args:
            patience (int): 許容されるエポック数で改善が見られない場合、訓練を停止する。
            verbose (bool): True の場合、各エポックの終わりにメッセージを出力する。
            delta (float): 前の最良の損失と比較して「改善」と見なされる最小の変化。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """検証損失が改善された場合、現在のモデルを保存します。"""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss
