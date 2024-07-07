import matplotlib.pyplot as plt
import numpy as np
import torch
from econml.metalearners import SLearner
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model, make_loader
from src.trainer import Trainer

# NNのランダム性を固定
torch.manual_seed(42)


# 評価
def get_roi(model, X_test):
    model.eval()
    with torch.no_grad():
        # 1000個ずつに分けて推論
        for i in range(0, len(X_test), 1000):
            X_test_batch = torch.tensor(X_test[i : i + 1000], dtype=torch.float32)
            q_test_batch = model(X_test_batch)["pred"]
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
    plt.savefig("loss.png")


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
    num_epochs = 150
    lr = 0.0001
    std = 1.0
    model_name = "Direct"
    batch_size = 128
    dataset = DatasetGenerator(n_samples, n_features, std, seed)
    dataset = dataset.generate_dataset()
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
    model_params = {"input_dim": n_features}
    model = get_model(model_name=model_name, model_params=model_params)
    roi_dic = {}
    trainer = Trainer(num_epochs=num_epochs, lr=lr)
    model = trainer.train(train_dl=train_dl, val_dl=val_dl, model=model)
    roi = get_roi(model, X_test)
    roi_dic["DR"] = roi
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
