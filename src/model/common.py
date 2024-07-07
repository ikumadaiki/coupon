from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from src.model.direct import (
    DirectCollator,
    DirectNonLinear,
    TestDirectDataset,
    TrainDirectDataset,
)
from src.model.slearner import SlearnerDataset, SLearnerNonLinear

# NNのランダム性を固定
torch.manual_seed(42)


def make_loader(
    dataset: dict[str, NDArray[Any]],
    model_name: str,
    batch_size: int,
    train_flg: bool,
    seed: int,
) -> DataLoader:
    collator = None
    if model_name == "SLearner":
        ds = SlearnerDataset(
            X=dataset["features"],
            T=dataset["T"],
            y=dataset["y"],
            train_flg=train_flg,
            seed=seed,
        )
    elif model_name == "Direct":
        if train_flg:
            ds = TrainDirectDataset(
                X=dataset["features"],
                T=dataset["T"],
                y_r=dataset["y_r_dr"],
                y_c=dataset["y_c_dr"],
                seed=seed,
            )  # type: ignore
            collator = DirectCollator()
        else:
            # 1000個ずつdsに追加
            for i in range(0, len(dataset["features"]), 1000):
                if i == 0:
                    ds = TestDirectDataset(
                        X=dataset["features"][i : i + 1000],
                    )
                else:
                    ds += TestDirectDataset(
                        X=dataset["features"][i : i + 1000],
                    )

    if train_flg:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)  # type: ignore
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)  # type: ignore
    return dl


def get_model(model_name: str, model_params: dict) -> nn.Module:
    if model_name == "Direct":
        model = DirectNonLinear(**model_params)
    elif model_name == "SLearner":
        model = SLearnerNonLinear(**model_params)  # type: ignore
    return model
