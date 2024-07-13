from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


class DatasetGenerator:
    def __init__(
        self, n_samples: int, n_features: int, delta: float, predict_ps: bool, seed: int
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.delta = delta
        self.predict_ps = predict_ps
        self.seed = seed

    def generate_dataset(self) -> dict[str, NDArray[Any]]:
        dataset: Dict[str, NDArray[Any]] = {}
        dataset |= self.generate_feature()
        dataset |= self.generate_treatment(dataset["features"])
        auc_score = self.calculate_auc(dataset["T"], dataset["T_prob"])
        # import pdb; pdb.set_trace()
        if self.predict_ps:
            dataset |= self.predict_treatment(dataset["features"], dataset["T"])
            propensity_score = dataset["T_prob_pred"]
            rmse = self.calculate_rmse(dataset["T_prob"], dataset["T_prob_pred"])
        else:
            propensity_score = dataset["T_prob"]
        dataset |= self.generate_visit(dataset["features"], dataset["T"])
        dataset |= self.generate_conversion(
            dataset["features"], dataset["T"], dataset["y_c"]
        )
        dataset |= self.culculate_doubly_robust(
            dataset["features"],
            dataset["T"],
            propensity_score,
            dataset["y_r"],
            dataset["y_c"],
        )
        dataset |= self.culculate_ipw(
            dataset["T"], propensity_score, dataset["y_r"], dataset["y_c"]
        )

        return dataset

    def generate_feature(self) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        features = np.random.normal(size=(self.n_samples, self.n_features))
        return {"features": features}

    def generate_treatment(self, features: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        logistic_model = LogisticRegression(max_iter=1000)
        target = (
            np.dot(features, np.random.uniform(0.1, 0.5, size=features.shape[1]))
            - 0.5
            + np.random.normal(0, 0.5, size=len(features))
            > 0
        ).astype(int)
        logistic_model.fit(features, target)
        T_prob = logistic_model.predict_proba(features)[:, 1]
        # T_prob = sigmoid(
        #     np.dot(features, np.random.uniform(0.1, 0.5, size=features.shape[1]))
        #     - 0.5
        #     + np.random.normal(0, 0.5, size=len(features))
        # )
        T_prob = T_prob.clip(0.01, 0.99)
        T: NDArray[Any] = np.random.binomial(1, T_prob).astype(bool)
        treatment_prob = T_prob[T == 1]
        control_prob = T_prob[T == 0]
        plt.hist(treatment_prob, bins=20, alpha=0.5, label="T=1")
        plt.hist(control_prob, bins=20, alpha=0.5, label="T=0")
        plt.legend()
        plt.savefig("treatment_prob.png")

        # import pdb

        # pdb.set_trace()
        return {"T": T, "T_prob": T_prob}

    # T_Probを予測値に変更
    def predict_treatment(
        self,
        features: NDArray[Any],
        T: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        lgb = LGBMClassifier(verbose=-1, random_state=42)
        lgb.fit(features, T)
        T_prob_pred = lgb.predict_proba(features)[:, 1]
        T_prob_pred = T_prob_pred.clip(0.01, 0.99)
        return {"T_prob_pred": T_prob_pred}

    # T_Prob_predのAUCを計算
    def calculate_auc(
        self,
        T: NDArray[Any],
        T_prob_pred: NDArray[Any],
    ) -> float:
        return float(roc_auc_score(T, T_prob_pred))

    # T_ProbとT_Prob_predのRMSEを計算
    def calculate_rmse(
        self,
        T_prob: NDArray[Any],
        T_prob_pred: NDArray[Any],
    ) -> float:
        return float(np.sqrt(np.mean((T_prob - T_prob_pred) ** 2)))

    def generate_visit(
        self,
        features: NDArray[Any],
        T: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        interaction_effects = np.exp(features[:, 0] + features[:, 3]) * 0.2
        baseline_effect = 0.2 + features[:, 2] + features[:, 4]
        treatment_effect = T * interaction_effects
        std = self.delta * np.sqrt(np.pi / 2)
        noise = np.random.normal(0, std, size=len(features))
        prob_visit = np.clip(
            sigmoid(baseline_effect + treatment_effect) + noise,
            0.05,
            0.95,
        )
        visit = np.random.binomial(1, prob_visit)
        plt.clf()
        plt.hist(prob_visit, bins=20, alpha=0.5, label="Visit")
        plt.savefig("visit_prob.png")
        return {"y_c": visit}

    def generate_conversion(
        self,
        features: NDArray[Any],
        T: NDArray[Any],
        visit: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        noise = np.random.normal(0, self.delta, size=len(features))
        interaction_effects_purchase = np.exp(features[:, 1] + features[:, 6]) * 0.2
        baseline_effect_purchase = 0.1 + features[:, 5] + features[:, 7]
        treatment_effect_purchase = T * interaction_effects_purchase
        std = self.delta * np.sqrt(np.pi / 2)
        noise = np.random.normal(0, std, size=len(features))
        prob_purchase = np.clip(
            sigmoid(baseline_effect_purchase + treatment_effect_purchase) + noise,
            0.10,
            0.90,
        )
        purchase = np.where(visit == 1, np.random.binomial(1, prob_purchase), 0)
        plt.clf()
        plt.hist(prob_purchase, bins=20, alpha=0.5, label="Purchase")
        plt.savefig("purchase_prob.png")
        # import pdb; pdb.set_trace()
        return {"y_r": purchase}

    def culculate_doubly_robust(
        self,
        features: NDArray[Any],
        T: NDArray[Any],
        T_prob: NDArray[Any],
        y_r: NDArray[Any],
        y_c: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        # y_rとy_cの期待値を予測するモデルを学習
        treatment_mask = T == 1
        control_mask = T == 0
        treatment_features = features[treatment_mask]
        control_features = features[control_mask]
        treatment_purchase = y_r[treatment_mask]
        control_purchase = y_r[control_mask]
        treatment_visit = y_c[treatment_mask]
        control_visit = y_c[control_mask]

        mu_r_0 = LGBMClassifier(verbose=-1, random_state=42).fit(
            control_features, control_purchase
        )
        mu_r_1 = LGBMClassifier(verbose=-1, random_state=42).fit(
            treatment_features, treatment_purchase
        )
        mu_c_0 = LGBMClassifier(verbose=-1, random_state=42).fit(
            control_features, control_visit
        )
        mu_c_1 = LGBMClassifier(verbose=-1, random_state=42).fit(
            treatment_features, treatment_visit
        )

        doubly_robust = {}
        doubly_robust["y_r_dr"] = np.where(
            T == 1,
            (y_r - mu_r_1.predict_proba(features)[:, 1]) / T_prob
            + mu_r_1.predict_proba(features)[:, 1],
            (y_r - mu_r_0.predict_proba(features)[:, 1]) / (1 - T_prob)
            + mu_r_0.predict_proba(features)[:, 1],
        )
        doubly_robust["y_c_dr"] = np.where(
            T == 1,
            (y_c - mu_c_1.predict(features)) / T_prob + mu_c_1.predict(features),
            (y_c - mu_c_0.predict(features)) / (1 - T_prob) + mu_c_0.predict(features),
        )
        return doubly_robust

    def culculate_ipw(
        self,
        T: NDArray[Any],
        T_prob: NDArray[Any],
        y_r: NDArray[Any],
        y_c: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        ipw = {}

        ipw["y_r_ipw"] = np.where(
            T == 1,
            y_r / T_prob,
            y_r / (1 - T_prob),
        )
        ipw["y_c_ipw"] = np.where(
            T == 1,
            y_c / T_prob,
            y_c / (1 - T_prob),
        )
        return ipw


def split_dataset(
    dataset: Dict[str, NDArray[Any]],
) -> Tuple[Dict[str, NDArray[Any]], Dict[str, NDArray[Any]], Dict[str, NDArray[Any]]]:
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset["features"])),
        train_size=0.8,
        random_state=42,
        stratify=dataset["T"],
    )
    train_val_dataset = {}
    test_dataset = {}
    for key, value in dataset.items():
        train_val_dataset[key] = value[train_val_idx]
        test_dataset[key] = value[test_idx]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_dataset["features"])),
        train_size=0.75,
        random_state=42,
        stratify=train_val_dataset["T"],
    )
    train_dataset = {}
    val_dataset = {}
    for key, value in train_val_dataset.items():
        train_dataset[key] = value[train_idx]
        val_dataset[key] = value[val_idx]

    return train_dataset, val_dataset, test_dataset
