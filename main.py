import matplotlib.pyplot as plt
import torch
from econml.metalearners import SLearner
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

from src.evaluate.evaluate import calculate_values
from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model, make_loader
from src.trainer import Trainer

# NNのランダム性を固定
torch.manual_seed(42)


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


def cost_curve(incremental_costs, incremental_values):
    plt.plot(
        incremental_costs / incremental_costs.max(),
        incremental_values / incremental_values.max(),
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
    batch_size = 128
    model_name = "Direct"
    model_params = {"input_dim": n_features}
    dataset = DatasetGenerator(n_samples, n_features, std, seed)
    dataset = dataset.generate_dataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
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
    test_dl = make_loader(
        test_dataset,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=False,
        seed=seed,
    )
    model = get_model(model_name=model_name, model_params=model_params)
    roi_dic = {}
    trainer = Trainer(num_epochs=num_epochs, lr=lr)
    model = trainer.train(train_dl=train_dl, val_dl=val_dl, model=model)
    trainer.save_model(model, "model.pth")
    predictions = trainer.predict(test_dl, model).squeeze()
    roi_dic["DR"] = predictions
    roi_tpmsl = get_roi_tpmsl(
        train_dataset["features"],
        train_dataset["y_r"],
        train_dataset["y_c"],
        train_dataset["T"],
        test_dataset["features"],
    )
    roi_dic["TPMSL"] = roi_tpmsl
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], test_dataset["T"], test_dataset["y_r"], test_dataset["y_c"]
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
    plt.savefig("cost_curve_.png")


if __name__ == "__main__":
    main(predict_ps=True)
