from src.model.common import get_model, make_loader
from src.preprocess.make_data import DatasetGenerator, split_dataset
from src.trainer import Trainer


def main() -> None:
    seed = 42
    n_samples = 100_000
    n_features = 8
    std = 1.0
    num_epochs = 200
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
    import pdb

    pdb.set_trace()
    # val_dl = make_loader(
    #     val_dataset,
    #     model_name=model_name,
    #     batch_size=batch_size,
    #     train_flg=,
    #     seed=seed,
    # )
    model = get_model(model_name=model_name, model_params=model_params)
    trainer = Trainer(num_epochs=num_epochs, lr=lr)
    model = trainer.train(train_dl=train_dl, model=model)

    # modelをsaveする
    trainer.save_model(model, "model.pth")

    # plt.clf()
    # for roi in roi_dic:
    #     incremental_costs, incremental_values = calculate_values(
    #         roi_dic[roi], T_test, y_r_test, y_c_test
    #     )
    #     plt.plot(
    #         incremental_costs / max(incremental_costs),
    #         incremental_values / max(incremental_values),
    #         label=roi,
    #     )
    # plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    # plt.xlabel("Incremental Costs")
    # plt.ylabel("Incremental Values")
    # plt.legend()
    # plt.savefig("cost_curve.png")


if __name__ == "__main__":
    main()
