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
        self.treatment_idx_to_control_idx = {}
        # unused_control_idx = set(list(range(len(self.X_control))))
        self.ratio = round(len(self.X_control) / len(self.X_treated))
        ununsed_control_idx = list(range(len(self.X_control)))
        for i in range(len(self.X_treated)):
            control_idx_i = np.random.choice(
                ununsed_control_idx,
                self.ratio,
                replace=False,
            )
            self.treatment_idx_to_control_idx[i] = control_idx_i
            # unused_control_idx -= set(control_idx_i)

    def __len__(self) -> int:
        return len(self.X_treated)

    def __getitem__(
        self, idx: int
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], int]:
        # treatmentのデータを取得
        X_treated = self.X_treated[idx]
        y_r_treated = self.y_r_treated[idx]
        y_c_treated = self.y_c_treated[idx]

        # controlのデータを取得
        control_idx = self.treatment_idx_to_control_idx[idx]
        X_control = self.X_control[control_idx]
        y_r_control = self.y_r_control[control_idx]
        y_c_control = self.y_c_control[control_idx]

        # X = np.concatenate([X_treated, X_control], axis=0)
        # T = np.concatenate([np.ones(len(X_treated)), np.zeros(len(X_control))], axis=0)
        # y_r = np.concatenate([y_r_treated, y_r_control], axis=0)
        # y_c = np.concatenate([y_c_treated, y_c_control], axis=0)

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control, self.ratio


class DirectCollator:
    def __call__(self, batch: list[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # バッチを作成
        # X = torch.cat([torch.tensor(x[0], dtype=torch.float32) for x in batch], dim=0)
        # T = torch.cat([torch.tensor(x[1], dtype=torch.float32) for x in batch], dim=0)
        # y_r = torch.cat([torch.tensor(x[2], dtype=torch.float32) for x in batch], dim=0)
        # y_c = torch.cat([torch.tensor(x[3], dtype=torch.float32) for x in batch], dim=0)
        
        X_treated = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        y_r_treated = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        y_c_treated = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        X_control = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        y_r_control = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        y_c_control = torch.tensor([x[5] for x in batch], dtype=torch.float32)

        N, D = X_treated.shape
        ratio = batch[0][6]
        X_control = X_control.reshape(N * ratio, D)
        y_r_control = y_r_control.reshape(N * ratio, 1)
        y_c_control = y_c_control.reshape(N * ratio, 1)
        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control


        # return {
        #     "X": X,
        #     "T": T,
        #     "y_r": y_r,
        #     "y_c": y_c,
        # }


class TestDirectDataset(Dataset):  # type: ignore
    def __init__(
        self,
        X: NDArray[Any],
    ):
        self.X = torch.tensor(X).to(dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"X": self.X[idx]}


# 非線形モデルの定義
class DirectNonLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(DirectNonLinear, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.fc2 = nn.Linear(2 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, int(0.5 * input_dim))
        self.fc4 = nn.Linear(int(0.5 * input_dim), 1)

    def forward(
        self,
        X: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        y_r: Optional[torch.Tensor] = None,
        y_c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred = self._predict(X)
        if T is not None and y_r is not None and y_c is not None:
            treated_mask = T == 1
            X_treated = X[treated_mask]
            y_r_treated = y_r[treated_mask]
            y_c_treated = y_c[treated_mask]
            q_treated = self._predict(X_treated)

            control_mask = T == 0
            X_control = X[control_mask]
            y_r_control = y_r[control_mask]
            y_c_control = y_c[control_mask]
            q_control = self._predict(X_control)

            loss_1 = custom_loss(y_r_treated, y_c_treated, q_treated, X_treated.size(0))
            loss_0 = custom_loss(y_r_control, y_c_control, q_control, X_control.size(0))
            loss = loss_1 - loss_0

            return {"pred": pred, "loss": loss}
        else:
            return {"pred": pred}

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


# 損失関数の定義
def custom_loss(
    y_r: torch.Tensor, y_c: torch.Tensor, q: torch.Tensor, group_size: int
) -> torch.Tensor:
    q = torch.clamp(q, 1e-6, 1 - 1e-6)
    logit_q = torch.log(q / (1 - q))
    loss = -torch.sum(y_r * logit_q + y_c * torch.log(1 - q)) / group_size
    return loss


# def plot_loss(loss_history: list, loss_history_val: list) -> None:
#     plt.plot(loss_history, label="Train")
#     plt.plot(loss_history_val, label="Validation")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig("loss.png")
