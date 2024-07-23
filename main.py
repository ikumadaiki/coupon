from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from src.evaluate.evaluate import calculate_values, cost_curve
from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model, make_loader
from src.trainer import Trainer

# NNのランダム性を固定
torch.manual_seed(42)


def get_roi_tpmsl(
    train_dataset: dict[str, NDArray[Any]], test_dataset: dict[str, NDArray[Any]]
) -> NDArray[np.float64]:
    X = np.concatenate(
        [train_dataset["features"], train_dataset["T"].reshape(-1, 1)], axis=1
    )
    reg_r = LGBMClassifier(verbose=-1, random_state=42)
    reg_r.fit(X, train_dataset["y_r"])
    reg_c = LGBMClassifier(verbose=-1, random_state=42)
    reg_c.fit(X, train_dataset["y_c"])
    X_0 = np.hstack(
        [test_dataset["features"], np.zeros((len(test_dataset["features"]), 1))]
    )
    X_1 = np.hstack(
        [test_dataset["features"], np.ones((len(test_dataset["features"]), 1))]
    )
    mu_r_0 = reg_r.predict_proba(X_0)[:, 1]
    mu_r_1 = reg_r.predict_proba(X_1)[:, 1]
    mu_c_0 = reg_c.predict_proba(X_0)[:, 1]
    mu_c_1 = reg_c.predict_proba(X_1)[:, 1]
    tau_r = mu_r_1 - mu_r_0
    tau_c = mu_c_1 - mu_c_0
    rmse_mu_r_0 = np.sqrt(np.mean((test_dataset["true_mu_r_0"] - mu_r_0) ** 2))
    rmse_mu_r_1 = np.sqrt(np.mean((test_dataset["true_mu_r_1"] - mu_r_1) ** 2))
    rmse_mu_c_0 = np.sqrt(np.mean((test_dataset["true_mu_c_0"] - mu_c_0) ** 2))
    rmse_mu_c_1 = np.sqrt(np.mean((test_dataset["true_mu_c_1"] - mu_c_1) ** 2))
    rmse_tau_r = np.sqrt(np.mean((test_dataset["true_tau_r"] - tau_r) ** 2))
    rmse_tau_c = np.sqrt(np.mean((test_dataset["true_tau_c"] - tau_c) ** 2))
    # AUCを計算
    auc_mu_r_0 = roc_auc_score(
        np.round(np.clip(test_dataset["true_mu_r_0"], 0, 1)), mu_r_0
    )
    auc_mu_r_1 = roc_auc_score(
        np.round(np.clip(test_dataset["true_mu_r_1"], 0, 1)), mu_r_1
    )
    auc_mu_c_0 = roc_auc_score(
        np.round(np.clip(test_dataset["true_mu_c_0"], 0, 1)), mu_c_0
    )
    auc_mu_c_1 = roc_auc_score(
        np.round(np.clip(test_dataset["true_mu_c_1"], 0, 1)), mu_c_1
    )

    roi_tpmsl = tau_r / tau_c
    scaler = MinMaxScaler()
    roi_tpmsl = scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()
    rmse_roi = np.sqrt(np.mean((test_dataset["true_ROI"] - roi_tpmsl) ** 2))
    # import pdb

    # pdb.set_trace()
    return roi_tpmsl


def main(predict_ps: bool) -> None:
    seed = 42
    n_samples = 50_000
    n_features = 4
    num_epochs = 50
    lr = 0.001
    delta = 0.0
    batch_size = 512
    model_name = "Direct"
    model_params = {"input_dim": n_features}
    dataset = DatasetGenerator(
        n_samples, n_features, delta, predict_ps=predict_ps, only_rct=False, seed=seed
    )
    dataset = dataset.generate_dataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    model = get_model(model_name=model_name, model_params=model_params)
    method_list: list = ["DR", "Direct", "IPW"]
    method_list = method_list[:2]
    # method_list = []
    roi_dic = {}
    for i, method in enumerate(method_list):
        train_dl = make_loader(
            train_dataset,
            model_name=model_name,
            batch_size=batch_size,
            train_flg=True,
            method=method,
            seed=seed,
        )
        val_dl = make_loader(
            val_dataset,
            model_name=model_name,
            batch_size=batch_size,
            train_flg=True,
            method=method,
            seed=seed,
        )
        test_dl = make_loader(
            test_dataset,
            model_name=model_name,
            batch_size=batch_size,
            train_flg=False,
            method=method,
            seed=seed,
        )
        trainer = Trainer(num_epochs=num_epochs, lr=lr)
        model = trainer.train(train_dl=train_dl, val_dl=val_dl, model=model)
        trainer.save_model(model, "model.pth")
        predictions = trainer.predict(dl=test_dl, model=model).squeeze()
        roi_dic[method] = predictions
    roi_tpmsl = get_roi_tpmsl(
        train_dataset,
        test_dataset,
    )
    roi_dic["TPMSL_LGBM"] = roi_tpmsl
    roi_dic["Optimal"] = test_dataset["true_tau_r"] / test_dataset["true_tau_c"]
    dataset_only_rct = DatasetGenerator(
        n_samples, n_features, delta, predict_ps=predict_ps, only_rct=True, seed=seed
    )
    dataset_only_rct = dataset_only_rct.generate_dataset()
    train_dataset_only_RCT, val_dataset_only_RCT, test_dataset_only_RCT = split_dataset(dataset_only_rct)
    train_dl_only_RCT = make_loader(
        train_dataset_only_RCT,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=True,
        method="Direct",
        seed=seed,
    )
    val_dl_only_RCT = make_loader(
        val_dataset_only_RCT,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=True,
        method="Direct",
        seed=seed,
    )
    test_dl_only_RCT = make_loader(
        test_dataset_only_RCT,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=False,
        method="Direct",
        seed=seed,
    )
    trainer = Trainer(num_epochs=num_epochs, lr=lr)
    model = trainer.train(train_dl=train_dl_only_RCT, val_dl=val_dl_only_RCT, model=model)
    trainer.save_model(model, "model.pth")
    predictions = trainer.predict(dl=test_dl_only_RCT, model=model).squeeze()
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], test_dataset["true_tau_r"], test_dataset["true_tau_c"]
        )
        cost_curve(incremental_costs, incremental_values, label=roi)
    incremental_costs_only_RCT, incremental_values_only_RCT = calculate_values(
        predictions, test_dataset_only_RCT["true_tau_r"], test_dataset_only_RCT["true_tau_c"]
    )
    cost_curve(incremental_costs_only_RCT, incremental_values_only_RCT, label="Direct_only_RCT")


if __name__ == "__main__":
    main(predict_ps=True)
