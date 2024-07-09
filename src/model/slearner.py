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
            T = torch.tensor(self.T[idx], dtype=torch.float32)
            X = torch.cat([X, T])
        data = {"X": X}
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            data["y"] = y
        return data


# 非線形モデルの定義
class SLearnerNonLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(SLearnerNonLinear, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.fc2 = nn.Linear(2 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, int(0.5 * input_dim))
        self.fc4 = nn.Linear(int(0.5 * input_dim), 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> dict:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        if y is not None:
            return {"pred": x, "loss": self.criterion(x, y)}
        else:
            return {"pred": x}
