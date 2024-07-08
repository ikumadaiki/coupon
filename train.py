import matplotlib.pyplot as plt

from src.evaluate.evaluate import calculate_values, cost_curve
from src.make_data import DatasetGenerator, split_dataset
from src.model.common import get_model, make_loader
from src.trainer import Trainer


def train() -> None:
    seed = 42
    n_samples = 100_000
    n_features = 8
    std = 1.0
    num_epochs = 100
    lr = 0.0001
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
    trainer = Trainer(num_epochs=num_epochs, lr=lr)
    model = trainer.train(train_dl=train_dl, val_dl=val_dl, model=model)
    trainer.save_model(model, "model.pth")
    predictions = trainer.predict(dl=test_dl, model=model).squeeze()
    # cost_curve(incremental_costs, incremental_values)
    roi_dic = {}
    roi_dic["DR"] = predictions
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], test_dataset["T"], test_dataset["y_r"], test_dataset["y_c"]
        )
        cost_curve(incremental_costs, incremental_values)


if __name__ == "__main__":
    train()
