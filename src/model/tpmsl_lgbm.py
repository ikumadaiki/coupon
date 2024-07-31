from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def get_roi_tpmsl(
    train_dataset: dict[str, NDArray[Any]], test_dataset: dict[str, NDArray[Any]]
) -> NDArray[Any]:
    X = np.concatenate(
        [train_dataset["features"], train_dataset["T"].reshape(-1, 1)], axis=1
    )
    reg_r = LGBMClassifier(verbose=-1, random_state=0)
    reg_r.fit(X, train_dataset["y_r"])
    reg_c = LGBMClassifier(verbose=-1, random_state=0)
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
    # auc_mu_r_0 = roc_auc_score(
    #     np.round(np.clip(test_dataset["true_mu_r_0"], 0, 1)), mu_r_0
    # )
    # auc_mu_r_1 = roc_auc_score(
    #     np.round(np.clip(test_dataset["true_mu_r_1"], 0, 1)), mu_r_1
    # )
    # auc_mu_c_0 = roc_auc_score(
    #     np.round(np.clip(test_dataset["true_mu_c_0"], 0, 1)), mu_c_0
    # )
    # auc_mu_c_1 = roc_auc_score(
    #     np.round(np.clip(test_dataset["true_mu_c_1"], 0, 1)), mu_c_1
    # )

    roi_tpmsl = tau_r / tau_c
    scaler = MinMaxScaler()
    roi_tpmsl = scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()
    rmse_roi = np.sqrt(np.mean((test_dataset["true_ROI"] - roi_tpmsl) ** 2))
    # import pdb

    # pdb.set_trace()
    return roi_tpmsl


# model_name = "SLearner"
# model_params = {"input_dim": n_features + 1}
# model = get_model(model_name=model_name, model_params=model_params)
# method_list = ["cost", "revenue"]
# method_list = []
# predictions_sl = {}
# for method in method_list:
#     train_dl = make_loader(
#         train_dataset,
#         model_name=model_name,
#         batch_size=batch_size,
#         train_flg=True,
#         method=method,
#         seed=seed,
#     )
#     val_dl = make_loader(
#         val_dataset,
#         model_name=model_name,
#         batch_size=batch_size,
#         train_flg=True,
#         method=method,
#         seed=seed,
#     )

#     trainer = Trainer(num_epochs=num_epochs_list[2], lr=lr_list[2])
#     model = trainer.train(
#         train_dl=train_dl, val_dl=val_dl, model=model, method=method
#     )
#     trainer.save_model(model, f"model_{method}.pth")
#     predictions = trainer.predict(dl=test_dl, model=model).squeeze()
#     predictions_sl[method] = predictions
# # roi_dic["TPMSL_NN"] = predictions_sl["revenue"] / predictions_sl["cost"]
