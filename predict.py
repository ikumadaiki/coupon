import numpy as np
import torch
from econml.metalearners import SLearner
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from src.prediction.model_direct import NonLinearModel

# NNのランダム性を固定
torch.manual_seed(42)


# 評価
def get_roi(model: NonLinearModel, X_test: NDArray[np.float_]) -> NDArray[np.float_]:
    model.eval()
    with torch.no_grad():
        # 1000個ずつに分けて推論
        for i in range(0, len(X_test), 1000):
            X_test_batch = torch.tensor(X_test[i : i + 1000], dtype=torch.float32)
            q_test_batch = model(X_test_batch)
            if i == 0:
                q_test = q_test_batch
            else:
                q_test = torch.cat([q_test, q_test_batch], dim=0)
        roi_direct = q_test.numpy()
        roi_direct = roi_direct.reshape(1, -1)[0]
        return roi_direct


def get_roi_tpmsl(
    X_train: NDArray[np.float_],
    y_r_train: NDArray[np.float_],
    y_c_train: NDArray[np.float_],
    T_train: NDArray[np.float_],
    X_test: NDArray[np.float_],
) -> NDArray[np.float_]:
    models = LGBMRegressor(verbose=-1)
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

    return roi_tpmsl
