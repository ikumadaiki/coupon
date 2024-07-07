import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from econml.metalearners import SLearner
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
from tqdm import tqdm

from src.make_data import DatasetGenerator, split_dataset
from src.model.common import make_loader

# NNのランダム性を固定
torch.manual_seed(42)


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


def main(predict_ps: bool) -> None:
    seed = 42
    n_samples = 100_000
    n_features = 8
    dic = {}
    num_epochs = 150
    lr = 0.0001
    std = 1.0
    model_name = "Direct"
    batch_size = 128
    dataset = DatasetGenerator(n_samples, n_features, std, seed)
    dataset = dataset.generate_dataset()
    dic = dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    X_train, X_val, X_test = (
        train_dataset["features"],
        val_dataset["features"],
        test_dataset["features"],
    )
    T_train, T_val, T_test = (
        train_dataset["T"],
        val_dataset["T"],
        test_dataset["T"],
    )
    y_r_train, y_r_val, y_r_test = (
        train_dataset["y_r"],
        val_dataset["y_r"],
        test_dataset["y_r"],
    )
    y_c_train, y_c_val, y_c_test = (
        train_dataset["y_c"],
        val_dataset["y_c"],
        test_dataset["y_c"],
    )
    y_r_ipw_train, y_r_ipw_val, y_r_ipw_test = (
        train_dataset["y_r_ipw"],
        val_dataset["y_r_ipw"],
        test_dataset["y_r_ipw"],
    )
    y_c_ipw_train, y_c_ipw_val, y_c_ipw_test = (
        train_dataset["y_c_ipw"],
        val_dataset["y_c_ipw"],
        test_dataset["y_c_ipw"],
    )
    y_r_dr_train, y_r_dr_val, y_r_dr_test = (
        train_dataset["y_r_dr"],
        val_dataset["y_r_dr"],
        test_dataset["y_r_dr"],
    )
    y_c_dr_train, y_c_dr_val, y_c_dr_test = (
        train_dataset["y_c_dr"],
        val_dataset["y_c_dr"],
        test_dataset["y_c_dr"],
    )
    train_dl = make_loader(
        train_dataset,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=True,
        seed=seed,
    )
    val_dl = make_loader(
        val_dataset,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=True,
        seed=seed,
    )

    method_dic = {
        # "Direct": [y_r_train, y_c_train, y_r_val, y_c_val],
        # "IPW": [y_r_ipw_train, y_c_ipw_train, y_r_ipw_val, y_c_ipw_val],
        "DR": [y_r_dr_train, y_c_dr_train, y_r_dr_val, y_c_dr_val]
    }
    roi_dic = {}
    for method in method_dic:
        dl, dl_val = train_dl, val_dl
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
