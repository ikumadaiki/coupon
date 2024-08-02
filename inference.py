import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.evaluate.evaluate import calculate_values, cost_curve, optimize_alpha
from src.make_data import DatasetGenerator
from src.model.common import get_model, make_loader
from src.model.tpmsl_lgbm import get_roi_tpmsl
from src.trainer import Trainer


def load_test_data(
    test_samples: int,
    n_features: int,
    delta: float,
    ps_delta: float,
    seed: int,
    model_name: str,
    batch_size: int,
    method: str,
) -> tuple:
    test_dataset = DatasetGenerator(
        test_samples,
        n_features,
        delta,
        ps_delta=ps_delta,
        predict_ps=False,
        only_rct=False,
        rct_ratio=0.0,
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
    return test_dataset, test_dl


def inference(
    n_features: int,
    ps_delta: float,
    rct_ratio: float,
    train_dataset: dict,
    test_dataset: dict,
    test_dl: DataLoader,
) -> None:
    roi_dic = {}
    method_list = ["DR", "Direct", "Direct_only_RCT"]
    method_list = ["DR"]
    for method in method_list:
        path = f"model_{method}.pth"
        # モデルの読み込み
        model = get_model(model_name="Direct", model_params={"input_dim": n_features})
        model.load_state_dict(torch.load(path))
        trainer = Trainer(num_epochs=10, lr=0.1)
        predictions = trainer.predict(dl=test_dl, model=model).squeeze()
        roi_dic[method] = predictions
        incremental_costs, incremental_values = calculate_values(
            predictions, test_dataset["true_tau_r"], test_dataset["true_tau_c"]
        )
        incremental_costs_alpha, incremental_values_alpha = optimize_alpha(
            rct_ratio,
            predictions,
            test_dataset["true_tau_r"],
            test_dataset["true_tau_c"],
        )
    roi_tpmsl = get_roi_tpmsl(
        train_dataset,
        test_dataset,
    )
    roi_dic["TPMSL_LGBM"] = roi_tpmsl
    roi_dic["Optimal"] = test_dataset["true_tau_r"] / test_dataset["true_tau_c"]
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], test_dataset["true_tau_r"], test_dataset["true_tau_c"]
        )
        cost_curve(rct_ratio, incremental_costs, incremental_values, label=roi, alpha=False)
    cost_curve(rct_ratio, incremental_costs_alpha, incremental_values_alpha, label="DR_alpha", alpha=True)
