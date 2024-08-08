import numpy as np
import torch

from inference import compare_alpha, inference, load_test_data
from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model
from src.trainer import Trainer

# NNのランダム性を固定
torch.manual_seed(42)


def main(predict_ps: bool, validate: bool) -> None:
    seed = 42
    n_samples = 10_000
    test_samples = 100_000
    n_features = 4
    delta = 0.0
    ps_delta = 0.0  # 4パターン
    alpha_list = []
    _alpha_list = [0.0, 0.025, 0.05, 0.075, 0.10]
    model_name = "Direct"
    model_params = (
        {"input_dim": n_features}
        if model_name == "Direct"
        else {"input_dim": n_features + 1}
    )
    method = "DR"
    only_rct = True if method == "Direct_only_RCT" else False
    num_epochs_list = [2000, 50]
    weight_decay_list = [1e-3]
    lr_list = [1e-3, 5e-3, 1e-2]
    batch_size_list = [2048]
    for rct_ratio in _alpha_list:
        alpha_list.append(rct_ratio)
        dataset = DatasetGenerator(
            n_samples=n_samples,
            n_features=n_features,
            delta=delta,
            ps_delta=ps_delta,
            predict_ps=predict_ps,
            only_rct=only_rct,
            rct_ratio=rct_ratio,
            train_flg=True,
            seed=seed,
        )
        dataset = dataset.generate_dataset()
        nan_mask = np.isnan(dataset["true_ROI"])
        dataset["true_ROI"][nan_mask] = 0.0
        train_dataset, val_dataset = split_dataset(dataset)
        model = get_model(model_name=model_name, model_params=model_params)
        test_dataset, test_dl = load_test_data(
            test_samples,
            n_features,
            delta,
            ps_delta=ps_delta,
            seed=seed,
            model_name=model_name,
            method=None,
        )
        trainer = Trainer(
            num_epochs=num_epochs_list[int(only_rct)],
            patience=10,
        )
        if not validate:
            model, _, best_lr, best_batch_size, weight_decay = trainer.grid_search(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model=model,
                lr_list=lr_list,
                weight_decay_list=weight_decay_list,
                batch_size_list=batch_size_list,
                model_name=model_name,
                method=method,
            )
            print(
                f"best_lr: {best_lr}, best_batch_size: {best_batch_size}, weight_decay: {weight_decay}"
            )
            trainer.save_model(model, f"model_{method}.pth")
            if method == "DR" or method == "IPW":
                trainer.save_model(model, f"model_{method}_{ps_delta}_{rct_ratio}.pth")

        inference(
            n_features=n_features,
            rct_ratio=rct_ratio,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            test_dl=test_dl,
        )

        compare_alpha(
            n_features=n_features,
            ps_delta=ps_delta,
            rct_ratio=rct_ratio,
            alpha_list=alpha_list,
            test_dataset=test_dataset,
            test_dl=test_dl,
        )


if __name__ == "__main__":
    main(predict_ps=True, validate=False)
