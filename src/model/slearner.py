from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# NNのランダム性を固定
torch.manual_seed(42)


class SlearnerDataset(Dataset):
    def __init__(
        self,
        X: NDArray[np.float_],
        T: NDArray[np.bool_],
        y: NDArray[np.float_],
        train_flg: bool,
        seed: int,
    ) -> None:
        np.random.seed(seed)
        self.X = X
        self.T = T
        self.y = y
        self.train_flg = train_flg

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
        Tuple[NDArray[np.float_], NDArray[np.float_]],
    ]:
        if self.train_flg:
            return self.X[idx], self.T[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]


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


def get_loss(
    num_epochs: int,
    lr: float,
    X_train: NDArray[np.float_],
    dl: DataLoader,
    dl_val: DataLoader,
) -> Tuple:
    model = NonLinearModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history, loss_history_val = [], []
    lambda_scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.90**epoch
    )
    criterion = nn.BCEWithLogitsLoss()

    # 学習ループ
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss, total_loss_val = 0, 0
        count_batches, count_batches_val = 0, 0

        average_loss = 0
        total = len(dl)
        desc = f"Epoch {epoch} AVG Loss: {average_loss:.4f}"
        for x, t, y in tqdm(dl, total=total, desc=desc, leave=False):
            optimizer.zero_grad()
            loss: torch.Tensor
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count_batches += 1

        average_loss = total_loss / count_batches
        loss_history.append(average_loss)
        lambda_scheduler.step()

        # 検証データでの損失関数の計算
        model.eval()
        with torch.no_grad():
            for x, t, y in tqdm(dl_val, total=total, desc=desc, leave=False):
                loss: torch.Tensor
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count_batches += 1

        average_loss_val = total_loss_val / count_batches_val
        loss_history_val.append(average_loss_val)

    return model, loss_history, loss_history_val


def plot_loss(loss_history: list, loss_history_val: list) -> None:
    plt.plot(loss_history, label="Train")
    plt.plot(loss_history_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
