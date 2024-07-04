from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def sigmoid(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return 1 / (1 + np.exp(-x))


def generate_feature(n: int, p: int, dic: Dict, seed: int) -> Dict:
    np.random.seed(seed)
    features = np.random.normal(size=(n, p))
    dic["features"] = features
    return dic


def generate_treatment(dic: Dict, seed: int) -> Dict:
    np.random.seed(seed)
    logistic_model = LogisticRegression(max_iter=1000)
    dic["target"] = (
        np.dot(
            dic["features"], np.random.uniform(0.1, 0.5, size=dic["features"].shape[1])
        )
        - 0.5
        + np.random.normal(0, 0.5, size=len(dic["features"]))
        > 0
    ).astype(int)
    logistic_model.fit(dic["features"], dic["target"])
    dic["T_prob"] = logistic_model.predict_proba(dic["features"])[:, 1]
    # dic["T_prob"] = sigmoid(
    #     np.dot(
    #         dic["features"], np.random.uniform(0.1, 0.5, size=dic["features"].shape[1])
    #     )
    #     - 0.5
    #     + np.random.normal(0, 0.5, size=len(dic["features"]))
    # )
    dic["T_prob"] = dic["T_prob"].clip(0.01, 0.99)
    dic["T"] = np.random.binomial(1, dic["T_prob"])
    if "target" in dic:
        dic.pop("target")
    # dic["T"]が1のdic["T_prob"]の分布を確認
    dic_1 = dic["T_prob"][dic["T"] == 1]
    dic_0 = dic["T_prob"][dic["T"] == 0]
    plt.hist(dic_1, bins=20, alpha=0.5, label="T=1")
    plt.hist(dic_0, bins=20, alpha=0.5, label="T=0")
    plt.legend()
    plt.savefig("treatment_prob.png")

    # import pdb

    # pdb.set_trace()
    return dic


# T_Probを予測値に変更
def predict_treatment(dic: Dict) -> Dict:
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(dic["features"], dic["T"])
    dic["T_prob"] = logistic_model.predict_proba(dic["features"])[:, 1]
    dic["T_prob"] = dic["T_prob"].clip(0.01, 0.99)
    return dic


def generate_visit(dic: Dict, std: float, seed: int) -> Dict:
    np.random.seed(seed)
    noise = np.random.normal(0, std, size=3 * len(dic["features"]))
    interaction_effects = sigmoid(np.sum(dic["features"], axis=1))
    baseline_effect = (
        0.3
        + dic["features"][:, 2] * 0.3
        + dic["features"][:, 4] * 0.1
        + noise[0 : len(dic["features"])]
    )
    treatment_effect = dic["T"] * (
        0.2
        + interaction_effects
        + noise[len(dic["features"]) : 2 * len(dic["features"])]
    )
    treatment_effect = np.clip(treatment_effect, 0.01, 100)
    prob_visit = np.clip(
        baseline_effect
        + treatment_effect
        + noise[2 * len(dic["features"]) : 3 * len((dic["features"]))],
        0.05,
        0.95,
    )
    dic["visit"] = np.random.binomial(1, prob_visit)
    return dic


def generate_conversion(dic: Dict, std: float, seed: int) -> Dict:
    np.random.seed(seed)
    noise = np.random.normal(0, std, size=3 * len(dic["features"]))
    interaction_effects_purchase = sigmoid(np.sum(dic["features"], axis=1))
    baseline_effect_purchase = (
        0.1
        + dic["features"][:, 5] * 0.2
        + dic["features"][:, 7] * 0.2
        + noise[0 : len(dic["features"])]
    )
    treatment_effect_purchase = dic["T"] * (
        0.1
        + interaction_effects_purchase
        + noise[len(dic["features"]) : 2 * len(dic["features"])]
    )
    treatment_effect_purchase = np.clip(treatment_effect_purchase, 0.01, 100)
    prob_purchase = np.clip(
        baseline_effect_purchase
        + noise[2 * len(dic["features"]) : 3 * len((dic["features"]))],
        0.10,
        0.90,
    )
    dic["purchase"] = np.where(
        dic["visit"] == 1, np.random.binomial(1, prob_purchase), 0
    )
    return dic


def predict_outcome(dic: Dict) -> Tuple:
    dic_t0 = {k: v[dic["T"] == 0] for k, v in dic.items()}
    dic_t1 = {k: v[dic["T"] == 1] for k, v in dic.items()}
    mu_r_0 = LGBMRegressor(verbose=-1).fit(dic_t0["features"], dic_t0["purchase"])
    mu_r_1 = LGBMRegressor(verbose=-1).fit(dic_t1["features"], dic_t1["purchase"])
    mu_c_0 = LGBMRegressor(verbose=-1).fit(dic_t0["features"], dic_t0["visit"])
    mu_c_1 = LGBMRegressor(verbose=-1).fit(dic_t1["features"], dic_t1["visit"])
    return mu_r_0, mu_r_1, mu_c_0, mu_c_1


def preprocess_data(
    dic: Dict,
    mu_r_0: LGBMRegressor,
    mu_r_1: LGBMRegressor,
    mu_c_0: LGBMRegressor,
    mu_c_1: LGBMRegressor,
) -> Dict:
    X, T, y_r, y_c, e = (
        dic["features"],
        dic["T"],
        dic["purchase"],
        dic["visit"],
        dic["T_prob"],
    )
    dic["y_r_ipw"] = np.where(T == 1, y_r / e, y_r / (1 - e))
    dic["y_c_ipw"] = np.where(T == 1, y_c / e, y_c / (1 - e))
    dic["y_r_dr"] = np.where(
        T == 1,
        (y_r - mu_r_1.predict(X)) / e + mu_r_1.predict(X),
        (y_r - mu_r_0.predict(X)) / (1 - e) + mu_r_0.predict(X),
    )
    dic["y_c_dr"] = np.where(
        T == 1,
        (y_c - mu_c_1.predict(X)) / e + mu_c_1.predict(X),
        (y_c - mu_c_0.predict(X)) / (1 - e) + mu_c_0.predict(X),
    )
    return dic


def split_data(dic: Dict, seed: int) -> Tuple:
    (
        X_train_val,
        X_test,
        T_train_val,
        T_test,
        y_r_train_val,
        y_r_test,
        y_c_train_val,
        y_c_test,
        y_r_ipw_train_val,
        y_r_ipw_test,
        y_c_ipw_train_val,
        y_c_ipw_test,
        y_r_dr_train_val,
        y_r_dr_test,
        y_c_dr_train_val,
        y_c_dr_test,
    ) = train_test_split(
        dic["features"],
        dic["T"],
        dic["purchase"],
        dic["visit"],
        dic["y_r_ipw"],
        dic["y_c_ipw"],
        dic["y_r_dr"],
        dic["y_c_dr"],
        train_size=0.8,
        random_state=42,
        stratify=dic["T"],
    )
    (
        X_train,
        X_val,
        T_train,
        T_val,
        y_r_train,
        y_r_val,
        y_c_train,
        y_c_val,
        y_r_ipw_train,
        y_r_ipw_val,
        y_c_ipw_train,
        y_c_ipw_val,
        y_r_dr_train,
        y_r_dr_val,
        y_c_dr_train,
        y_c_dr_val,
    ) = train_test_split(
        X_train_val,
        T_train_val,
        y_r_train_val,
        y_c_train_val,
        y_r_ipw_train_val,
        y_c_ipw_train_val,
        y_r_dr_train_val,
        y_c_dr_train_val,
        train_size=0.75,
        random_state=42,
        stratify=T_train_val,
    )
    return (
        X_train,
        X_val,
        X_test,
        T_train,
        T_val,
        T_test,
        y_r_train,
        y_r_val,
        y_r_test,
        y_c_train,
        y_c_val,
        y_c_test,
        y_r_ipw_train,
        y_r_ipw_val,
        y_c_ipw_train,
        y_c_ipw_val,
        y_r_dr_train,
        y_r_dr_val,
        y_c_dr_train,
        y_c_dr_val,
    )
