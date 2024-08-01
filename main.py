import torch

from inference import inference, load_test_data
from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model, make_loader
from src.trainer import Trainer

# NNのランダム性を固定
torch.manual_seed(42)


def main(predict_ps: bool, validate: bool) -> None:
    seed = 42
    n_samples = 100_000
    test_samples = 100_000
    n_features = 4
    delta = 0.0
    ps_delta = 0.3  # 4パターン
    rct_ratio = 0.05  # 8パターン # Direct:rct_ratioは大きい方がいい
    batch_size = 8
    weight_decay = 1e-2
    model_name = "Direct"
    model_params = {"input_dim": n_features}
    method = "DR"
    only_rct = True if method == "Direct_only_RCT" else False
    num_epochs_list = [50, 50]
    lr_list = [0.00001, 0.00001]
    dataset = DatasetGenerator(
        n_samples,
        n_features,
        delta,
        ps_delta=ps_delta,
        predict_ps=predict_ps,
        only_rct=only_rct,
        rct_ratio=rct_ratio,
        train_flg=True,
        seed=seed,
    )
    dataset = dataset.generate_dataset()
    train_dataset, val_dataset = split_dataset(dataset)
    model = get_model(model_name=model_name, model_params=model_params)
    test_dataset, test_dl = load_test_data(
        test_samples,
        n_features,
        delta,
        ps_delta=ps_delta,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        method=None,
    )
    if not validate:
        if only_rct:
            train_dl = make_loader(
                train_dataset,
                model_name=model_name,
                batch_size=batch_size,
                train_flg=True,
                method="Direct",
                seed=seed,
            )
            val_dl = make_loader(
                val_dataset,
                model_name=model_name,
                batch_size=batch_size,
                train_flg=True,
                method="Direct",
                seed=seed,
            )
        else:
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
    trainer = Trainer(
        num_epochs=num_epochs_list[int(only_rct)],
        lr=lr_list[int(only_rct)],
        weight_decay=weight_decay,
    )
    if not validate:
        model = trainer.train(
            train_dl=train_dl, val_dl=val_dl, model=model, method=method
        )
        trainer.save_model(model, f"model_{method}.pth")

    inference(
        n_features=n_features,
        ps_delta=ps_delta,
        rct_ratio=rct_ratio,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        test_dl=test_dl,
    )


if __name__ == "__main__":
    main(predict_ps=True, validate=True)
