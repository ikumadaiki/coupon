from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

# NNのランダム性を固定
torch.manual_seed(42)


class SlearnerDataset(Dataset):  # type: ignore
    def __init__(
        self,
        X: NDArray[Any],
        T: Optional[NDArray[Any]],
        y: Optional[NDArray[Any]],
        seed: int,
    ) -> None:
        np.random.seed(seed)
        self.X = X
        self.T = T
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.T is not None:
            T = torch.tensor([self.T[idx]], dtype=torch.float32)
            X = torch.cat([X, T], dim=0)

        data = {"X": X}
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            data["y"] = y
        return data


# 非線形モデルの定義
class SLearnerNonLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(SLearnerNonLinear, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim, int(0.5 * input_dim)),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(int(0.5 * input_dim), 1),  # Sigmoidを削除
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        pred = self.mlp(X).squeeze()
        if y is not None:
            return {"pred": pred, "loss": self.criterion(pred, y)}
        else:
            X_1 = torch.cat([X, torch.ones((X.size(0), 1))], dim=1)
            X_0 = torch.cat([X, torch.zeros((X.size(0), 1))], dim=1)
            mu_1 = self.mlp(X_1)
            mu_0 = self.mlp(X_0)
            tau = mu_1 - mu_0
            return {"pred": tau}
