import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

# NNのランダム性を固定
torch.manual_seed(42)


class TrainDirectDataset(Dataset):  # type: ignore
    def __init__(
        self,
        X: NDArray[Any],
        T: NDArray[Any],
        y_r: NDArray[Any],
        y_c: NDArray[Any],
        seed: int,
    ) -> None:
        np.random.seed(seed)
        self.X_treated = X[T == 1]
        self.y_r_treated = y_r[T == 1]
        self.y_c_treated = y_c[T == 1]
        self.X_control = X[T == 0]
        self.y_r_control = y_r[T == 0]
        self.y_c_control = y_c[T == 0]
        # teratment_idxからcontrol_idxをランダム二つ選択する辞書を作成
        self.control_idx_to_treatment_idx = {}
        # unused_control_idx = set(list(range(len(self.X_control))))
        self.ratio = math.ceil(len(self.X_control) / len(self.X_treated))
        self.ratio = 1
        # import pdb

        # pdb.set_trace()
        ununsed_treated_idx = list(range(len(self.X_treated)))
        for i in range(len(self.X_control)):
            treated_idx = np.random.choice(
                ununsed_treated_idx,
                self.ratio,
                replace=False,
            )
            self.control_idx_to_treatment_idx[i] = treated_idx
            # unused_control_idx -= set(control_idx_i)

    def __len__(self) -> int:
        return len(self.X_treated)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
    ]:
        if idx > len(self.X_control) - 1:
            idx = idx % len(self.X_control)
        X_control = self.X_control[idx]
        y_r_control = self.y_r_control[idx]
        y_c_control = self.y_c_control[idx]
        treated_idx = self.control_idx_to_treatment_idx[idx]
        X_treated = self.X_treated[treated_idx]
        y_r_treated = self.y_r_treated[treated_idx]
        y_c_treated = self.y_c_treated[treated_idx]

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control


class DirectCollator:
    def __call__(self, batch: list[Any]) -> Dict[str, torch.Tensor | int]:
        # バッチを作成
        X_treated = torch.tensor([x[0] for x in batch], dtype=torch.float32)  # (B, D)
        y_r_treated = torch.tensor(
            [x[1] for x in batch], dtype=torch.float32
        ).squeeze()  # (B, 1)
        y_c_treated = torch.tensor(
            [x[2] for x in batch], dtype=torch.float32
        ).squeeze()  # (B, 1)
        X_control = torch.tensor(
            [x[3] for x in batch], dtype=torch.float32
        )  # (B, 2, D)
        y_r_control = torch.tensor([x[4] for x in batch], dtype=torch.float32)  # (B, 2)
        y_c_control = torch.tensor([x[5] for x in batch], dtype=torch.float32)  # (B, 2)
        _, D = X_control.shape
        X_treated = X_treated.reshape(-1, D)  # (2B, D)
        y_r_treated = y_r_treated.reshape(-1, 1).squeeze()  # (2B, 1)
        y_c_treated = y_c_treated.reshape(-1, 1).squeeze()  # (2B, 1)
        X = torch.cat([X_treated, X_control], dim=0)
        treated_size = len(X_treated)
        y_r = torch.cat([y_r_treated, y_r_control], dim=0)
        y_c = torch.cat([y_c_treated, y_c_control], dim=0)

        return {
            "X": X,
            "treated_size": treated_size,
            "y_r": y_r,
            "y_c": y_c,
        }


class TestDirectDataset(Dataset):  # type: ignore
    def __init__(
        self,
        X: NDArray[Any],
    ):
        self.X = X.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, NDArray[Any]]:
        return {"X": self.X[idx]}


# 非線形モデルの定義
class DirectNonLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(DirectNonLinear, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(2 * input_dim, input_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(input_dim, int(0.5 * input_dim)),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        X: torch.Tensor,
        treated_size: Optional[torch.Tensor] = None,
        y_r: Optional[torch.Tensor] = None,
        y_c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred = self._predict(X)
        if treated_size is not None and y_r is not None and y_c is not None:
            q_treated = pred[:treated_size].squeeze()
            y_r_treated = y_r[:treated_size]
            y_c_treated = y_c[:treated_size]
            q_control = pred[treated_size:].squeeze()
            y_r_control = y_r[treated_size:]
            y_c_control = y_c[treated_size:]

            loss_1 = custom_loss(y_r_treated, y_c_treated, q_treated)

            loss_0 = custom_loss(y_r_control, y_c_control, q_control)

            loss = loss_1 - loss_0

            return {"pred": pred.detach(), "loss": loss}
        else:
            return {"pred": pred}

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # type: ignore


# 損失関数の定義
def custom_loss(y_r: torch.Tensor, y_c: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(q, 1e-2, 1 - 1e-2)
    logit_q = torch.log(q / (1 - q))

    loss = -torch.mean(y_r * logit_q + y_c * torch.log(1 - q))
    return loss
