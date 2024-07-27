import torch

from src.evaluate.evaluate import calculate_values, cost_curve
from src.make_data import DatasetGenerator
from src.model.common import get_model, make_loader
from src.trainer import Trainer


def inference() -> None:
    test_samples = 100_000
    n_features = 4
    delta = 0.0
    rct_ratio = 0.15
    seed = 42
    model_name = "Direct"
    batch_size = 256
    method = "DR"
    test_dataset = DatasetGenerator(
        test_samples,
        n_features,
        delta,
        predict_ps=False,
        only_rct=False,
        rct_ratio=rct_ratio,
        train_flg=False,
        seed=seed,
    )
    test_dataset = test_dataset.generate_dataset()
    test_dl = make_loader(
        test_dataset,
        model_name=model_name,
        batch_size=batch_size,
        train_flg=False,
        method=method,
        seed=seed,
    )

    path = f"model_{method}.pth"

    # モデルの読み込み
    model = get_model(model_name="Direct", model_params={"input_dim": n_features})

    model.load_state_dict(torch.load(path))

    trainer = Trainer(num_epochs=10, lr=0.1)
    predictions = trainer.predict(dl=test_dl, model=model).squeeze()
    incremental_costs, incremental_values = calculate_values(
        predictions, test_dataset["true_tau_r"], test_dataset["true_tau_c"]
    )
    cost_curve(incremental_costs, incremental_values, label=method)


if __name__ == "__main__":
    inference()
