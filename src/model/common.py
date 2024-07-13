from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

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
    train_flg: Optional[bool],
    method: Optional[str],
    seed: int,
) -> DataLoader:  # type: ignore
    collator = None
    ds: Dataset  # type: ignore
    if model_name == "SLearner":
        if method == "revenue":
            ds = SlearnerDataset(
                X=dataset["features"],
                T=dataset["T"],
                y=dataset["y_r"],
                seed=seed,
            )
        elif method == "cost":
            ds = SlearnerDataset(
                X=dataset["features"],
                T=dataset["T"],
                y=dataset["y_c"],
                seed=seed,
            )

    elif model_name == "Direct":
        if train_flg:
            if method == "DR":
                ds = TrainDirectDataset(
                    X=dataset["features"],
                    T=dataset["T"],
                    y_r=dataset["y_r_dr"],
                    y_c=dataset["y_c_dr"],
                    seed=seed,
                )  # type: ignore
            elif method == "IPW":
                ds = TrainDirectDataset(
                    X=dataset["features"],
                    T=dataset["T"],
                    y_r=dataset["y_r_ipw"],
                    y_c=dataset["y_c_ipw"],
                    seed=seed,
                )
            elif method == "Direct":
                ds = TrainDirectDataset(
                    X=dataset["features"],
                    T=dataset["T"],
                    y_r=dataset["y_r"],
                    y_c=dataset["y_c"],
                    seed=seed,
                )
            collator = DirectCollator()
        else:
            ds = TestDirectDataset(X=dataset["features"])

    if train_flg:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)  # type: ignore
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)  # type: ignore
    return dl


def get_model(model_name: str, model_params: Dict[str, int]) -> nn.Module:
    if model_name == "Direct":
        model = DirectNonLinear(**model_params)
    elif model_name == "SLearner":
        model = SLearnerNonLinear(**model_params)  # type: ignore
    return model
