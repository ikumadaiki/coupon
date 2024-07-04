from typing import Tuple

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


class CustomDataset(Dataset):
    def __init__(
        self,
        X_treated: NDArray[np.float_],
        y_r_treated: NDArray[np.float_],
        y_c_treated: NDArray[np.float_],
        X_control: NDArray[np.float_],
        y_r_control: NDArray[np.float_],
        y_c_control: NDArray[np.float_],
        seed: int,
    ) -> None:
        np.random.seed(seed)
        self.X_treated = X_treated
        self.y_r_treated = y_r_treated
        self.y_c_treated = y_c_treated
        self.X_control = X_control
        self.y_r_control = y_r_control
        self.y_c_control = y_c_control

        # teratment_idxからcontrol_idxをランダム二つ選択する辞書を作成
        self.treatment_idx_to_control_idx = {}
        # unused_control_idx = set(list(range(len(self.X_control))))
        ununsed_control_idx = list(range(len(self.X_control)))
        for i in range(len(self.X_treated)):
            control_idx_i = np.random.choice(
                ununsed_control_idx,
                round(len(X_control) / len(X_treated)),
                replace=False,
            )
            self.treatment_idx_to_control_idx[i] = control_idx_i
            # unused_control_idx -= set(control_idx_i)

    def __len__(self) -> int:
        return len(self.X_treated)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
    ]:
        # treatmentのデータを取得
        X_treated = self.X_treated[idx]
        y_r_treated = self.y_r_treated[idx]
        y_c_treated = self.y_c_treated[idx]

        # controlのデータを取得
        control_idx = self.treatment_idx_to_control_idx[idx]
        X_control = self.X_control[control_idx]
        y_r_control = self.y_r_control[control_idx]
        y_c_control = self.y_c_control[control_idx]

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control


class CustomCollator:
    def __call__(
        self, batch: list
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # バッチを作成
        X_treated = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        y_r_treated = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        y_c_treated = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        # controlは2つあるので、それぞれのデータを取得
        X_control = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        y_r_control = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        y_c_control = torch.tensor([x[5] for x in batch], dtype=torch.float32)

        # reshape
        N, D = X_treated.shape
        X_control = X_control.reshape(N * 2, D)
        y_r_control = y_r_control.reshape(N * 2, 1)
        y_c_control = y_c_control.reshape(N * 2, 1)

        return X_treated, y_r_treated, y_c_treated, X_control, y_r_control, y_c_control


def loader(
    X_train: NDArray[np.float_],
    T_train: NDArray[np.float_],
    y_r_train: NDArray[np.float_],
    y_c_train: NDArray[np.float_],
    X_val: NDArray[np.float_],
    T_val: NDArray[np.float_],
    y_r_val: NDArray[np.float_],
    y_c_val: NDArray[np.float_],
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    # treatmentとcontrolのデータを分割
    treatment_mask = T_train == 1
    X_train_treated = X_train[treatment_mask]
    y_r_train_treated = y_r_train[treatment_mask]
    y_c_train_treated = y_c_train[treatment_mask]

    control_mask = T_train == 0
    X_train_control = X_train[control_mask]
    y_r_train_control = y_r_train[control_mask]
    y_c_train_control = y_c_train[control_mask]

    treatment_mask_val = T_val == 1
    X_val_treated = X_val[treatment_mask_val]
    y_r_val_treated = y_r_val[treatment_mask_val]
    y_c_val_treated = y_c_val[treatment_mask_val]

    control_mask_val = T_val == 0
    X_val_control = X_val[control_mask_val]
    y_r_val_control = y_r_val[control_mask_val]
    y_c_val_control = y_c_val[control_mask_val]

    # データをテンソルに変換してDatasetを作成
    import pdb

    pdb.set_trace()
    ds = CustomDataset(
        X_train_treated,
        y_r_train_treated,
        y_c_train_treated,
        X_train_control,
        y_r_train_control,
        y_c_train_control,
        seed,
    )
    collator = CustomCollator()

    import pdb

    pdb.set_trace()
    # DataLoaderの定義
    dl = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collator)

    ds_val = CustomDataset(
        X_val_treated,
        y_r_val_treated,
        y_c_val_treated,
        X_val_control,
        y_r_val_control,
        y_c_val_control,
        seed,
    )
    collator_val = CustomCollator()
    dl_val = DataLoader(ds_val, batch_size=128, shuffle=True, collate_fn=collator_val)

    return dl, dl_val


# 非線形モデルの定義
class NonLinearModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.fc2 = nn.Linear(2 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, int(0.5 * input_dim))
        self.fc4 = nn.Linear(int(0.5 * input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    # 学習ループ
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss, total_loss_val = 0, 0
        count_batches, count_batches_val = 0, 0

        average_loss = 0
        total = len(dl)
        desc = f"Epoch {epoch} AVG Loss: {average_loss:.4f}"
        for x_1, y_r_1, y_c_1, x_0, y_r_0, y_c_0 in tqdm(
            dl, total=total, desc=desc, leave=False
        ):
            optimizer.zero_grad()
            q_1 = model(x_1)
            q_0 = model(x_0)
            loss_1 = custom_loss(y_r_1, y_c_1, q_1, x_1.size(0))
            loss_0 = custom_loss(y_r_0, y_c_0, q_0, x_0.size(0))
            loss = loss_1 - loss_0
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
            for x_1, y_r_1, y_c_1, x_0, y_r_0, y_c_0 in tqdm(
                dl_val, total=total, desc=desc, leave=False
            ):
                q_1 = model(x_1)
                q_0 = model(x_0)
                loss_1 = custom_loss(y_r_1, y_c_1, q_1, x_1.size(0))
                loss_0 = custom_loss(y_r_0, y_c_0, q_0, x_0.size(0))
                loss = loss_1 - loss_0
                total_loss_val += loss.item()
                count_batches_val += 1

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
