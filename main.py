import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from econml.metalearners import SLearner
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.make_data import DatasetGenerator

# NNのランダム性を固定
torch.manual_seed(42)


def split_data(dic, seed):
    (
        X_train_val,
        X_test,
        T_train_val,
        T_test,
        y_r_train_val,
        y_r_test,
        y_c_train_val,
        y_c_test,
        y_r_ipw_train_val,
        y_r_ipw_test,
        y_c_ipw_train_val,
        y_c_ipw_test,
        y_r_dr_train_val,
        y_r_dr_test,
        y_c_dr_train_val,
        y_c_dr_test,
    ) = train_test_split(
        dic["features"],
        dic["T"],
        dic["y_r"],
        dic["y_c"],
        dic["y_r_ipw"],
        dic["y_c_ipw"],
        dic["y_r_dr"],
        dic["y_c_dr"],
        train_size=0.8,
        random_state=42,
        stratify=dic["T"],
    )
    (
        X_train,
        X_val,
        T_train,
        T_val,
        y_r_train,
        y_r_val,
        y_c_train,
        y_c_val,
        y_r_ipw_train,
        y_r_ipw_val,
        y_c_ipw_train,
        y_c_ipw_val,
        y_r_dr_train,
        y_r_dr_val,
        y_c_dr_train,
        y_c_dr_val,
    ) = train_test_split(
        X_train_val,
        T_train_val,
        y_r_train_val,
        y_c_train_val,
        y_r_ipw_train_val,
        y_c_ipw_train_val,
        y_r_dr_train_val,
        y_c_dr_train_val,
        train_size=0.75,
        random_state=42,
        stratify=T_train_val,
    )
    return (
        X_train,
        X_val,
        X_test,
        T_train,
        T_val,
        T_test,
        y_r_train,
        y_r_val,
        y_r_test,
        y_c_train,
        y_c_val,
        y_c_test,
        y_r_ipw_train,
        y_r_ipw_val,
        y_c_ipw_train,
        y_c_ipw_val,
        y_r_dr_train,
        y_r_dr_val,
        y_c_dr_train,
        y_c_dr_val,
    )


class CustomDataset(Dataset):
    def __init__(
        self,
        X_treated,
        y_r_treated,
        y_c_treated,
        X_control,
        y_r_control,
        y_c_control,
        seed,
    ):
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

    def __len__(self):
        return len(self.X_treated)

    def __getitem__(self, idx):
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
    def __call__(self, batch):
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
    X_train, T_train, y_r_train, y_c_train, X_val, T_val, y_r_val, y_c_val, seed
):
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
    def __init__(self, input_dim):
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.fc2 = nn.Linear(2 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, int(0.5 * input_dim))
        self.fc4 = nn.Linear(int(0.5 * input_dim), 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# 損失関数の定義
def custom_loss(y_r, y_c, q, group_size):
    q = torch.clamp(q, 1e-6, 1 - 1e-6)
    logit_q = torch.log(q / (1 - q))
    loss = -torch.sum(y_r * logit_q + y_c * torch.log(1 - q)) / group_size
    return loss


def get_loss(num_epochs, lr, X_train, dl, dl_val):
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


# 評価
def get_roi(model, X_test):
    model.eval()
    with torch.no_grad():
        # 1000個ずつに分けて推論
        for i in range(0, len(X_test), 1000):
            X_test_batch = torch.tensor(X_test[i : i + 1000], dtype=torch.float32)
            q_test_batch = model(X_test_batch)
            if i == 0:
                q_test = q_test_batch
            else:
                q_test = torch.cat([q_test, q_test_batch], dim=0)
        roi_direct = q_test.numpy()
        roi_direct = roi_direct.reshape(1, -1)[0]
        return roi_direct


def plot_loss(loss_history, loss_history_val):
    plt.plot(loss_history, label="Train")
    plt.plot(loss_history_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def get_roi_tpmsl(X_train, y_r_train, y_c_train, T_train, X_test):
    models = LGBMRegressor()
    S_learner_r = SLearner(overall_model=models)
    S_learner_r.fit(y_r_train, T_train, X=X_train)
    S_learner_c = SLearner(overall_model=models)
    S_learner_c.fit(y_c_train, T_train, X=X_train)
    # 効果の推定
    tau_r = S_learner_r.effect(X_test)
    tau_c = S_learner_c.effect(X_test)
    roi_tpmsl = tau_r / tau_c
    scaler = MinMaxScaler()
    roi_tpmsl = scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()
    import pdb

    pdb.set_trace()
    return roi_tpmsl


def calculate_values(roi_scores, T_test, y_r_test, y_c_test):
    sorted_indices = np.argsort(roi_scores)[::-1]
    p_values = np.linspace(0, 1, 50)
    incremental_costs = []
    incremental_values = []

    for p in p_values:
        top_p_indices = sorted_indices[: int(p * len(roi_scores))]
        treatment_indices = T_test[top_p_indices] == 1

        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(y_r_test[top_p_indices][treatment_indices]) - np.mean(
            y_r_test[top_p_indices][~treatment_indices]
        )
        ATE_Yc = np.mean(y_c_test[top_p_indices][treatment_indices]) - np.mean(
            y_c_test[top_p_indices][~treatment_indices]
        )

        incremental_costs.append(ATE_Yc * np.sum(treatment_indices))
        incremental_values.append(ATE_Yr * np.sum(treatment_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
        incremental_costs[0] = 0
        incremental_values[0] = 0

    return incremental_costs, incremental_values


def cost_curve(incremental_costs, incremental_values):
    plt.plot(
        incremental_costs / max(incremental_costs),
        incremental_values / max(incremental_values),
    )
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.show()


def main(predict_ps=False):
    seed = 42
    n_samples = 100_000
    n_features = 8
    dic = {}
    num_epochs = 150
    lr = 0.0001
    std = 1.0
    dataset = DatasetGenerator(n_samples, n_features, std, seed)
    dataset = dataset.generate_dataset()
    dic = dataset
    (
        X_train,
        X_val,
        X_test,
        T_train,
        T_val,
        T_test,
        y_r_train,
        y_r_val,
        y_r_test,
        y_c_train,
        y_c_val,
        y_c_test,
        y_r_ipw_train,
        y_r_ipw_val,
        y_c_ipw_train,
        y_c_ipw_val,
        y_r_dr_train,
        y_r_dr_val,
        y_c_dr_train,
        y_c_dr_val,
    ) = split_data(dic, seed)
    method_dic = {
        # "Direct": [y_r_train, y_c_train, y_r_val, y_c_val],
        # "IPW": [y_r_ipw_train, y_c_ipw_train, y_r_ipw_val, y_c_ipw_val],
        "DR": [y_r_dr_train, y_c_dr_train, y_r_dr_val, y_c_dr_val]
    }
    roi_dic = {}
    for method in method_dic:
        dl, dl_val = loader(
            X_train,
            T_train,
            method_dic[method][0],
            method_dic[method][1],
            X_val,
            T_val,
            method_dic[method][2],
            method_dic[method][3],
            seed,
        )
        model, loss_history, loss_history_val = get_loss(
            num_epochs, lr, X_train, dl, dl_val
        )
        # plot_loss(loss_history, loss_history_val)
        roi = get_roi(model, X_test)
        roi_dic[method] = roi
    roi_tpmsl = get_roi_tpmsl(X_train, y_r_train, y_c_train, T_train, X_test)
    roi_dic["TPMSL"] = roi_tpmsl
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], T_test, y_r_test, y_c_test
        )
        plt.plot(
            incremental_costs / max(incremental_costs),
            incremental_values / max(incremental_values),
            label=roi,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.legend()
    plt.savefig("cost_curve.png")


if __name__ == "__main__":
    main(predict_ps=True)
