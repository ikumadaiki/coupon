import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.direct import DirectCollator, DirectNonLinear, TrainDirectDataset
from src.model.slearner import SlearnerDataset, SLearnerNonLinear


def make_loader(
    dataset: dict,
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
                y_r=dataset["purchase"],
                y_c=dataset["visit"],
                seed=seed,
            )  # type: ignore
        collator = DirectCollator()
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
