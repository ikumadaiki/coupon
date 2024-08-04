from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


class DatasetGenerator:
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        delta: float,
        ps_delta: float,
        predict_ps: bool,
        only_rct: bool,
        rct_ratio: float,
        train_flg: bool,
        seed: int,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.delta = delta
        self.ps_delta = ps_delta
        self.predict_ps = predict_ps
        self.only_rct = only_rct
        self.rct_ratio = rct_ratio
        self.train_flg = train_flg
        self.seed = seed

    def generate_dataset(self) -> dict[str, NDArray[Any]]:
        dataset: dict[str, NDArray[Any]] = {}
        dataset |= self.generate_feature()
        if self.train_flg:
            dataset |= self.set_rct_flag()
            dataset |= self.generate_treatment(dataset["features"], dataset["RCT_flag"])
            # auc_score = self.calculate_auc(dataset["T"], dataset["T_prob"])
            if self.predict_ps:
                dataset |= self.predict_treatment(
                    dataset["features"], dataset["T_prob"], dataset["RCT_flag"]
                )
                propensity_score = dataset["T_prob_pred"]
            else:
                propensity_score = dataset["T_prob"]
            rmse = self.calculate_rmse(dataset["T_prob"], propensity_score)
            print("RMSE: ", rmse)
            dataset |= self.generate_conversion(dataset["features"], dataset["T"])
            dataset |= self.generate_visit(
                dataset["features"], dataset["T"], dataset["y_r"]
            )
            dataset |= self.culculate_doubly_robust(
                dataset["features"],
                dataset["T"],
                propensity_score,
                dataset["y_r"],
                dataset["y_c"],
                dataset["true_mu_r_1"],
                dataset["true_mu_r_0"],
                dataset["true_mu_c_1"],
                dataset["true_mu_c_0"],
            )
            dataset |= self.culculate_ipw(
                dataset["T"], propensity_score, dataset["y_r"], dataset["y_c"]
            )
        else:
            dataset |= self.generate_conversion(dataset["features"])
            dataset |= self.generate_visit(dataset["features"])
        dataset["true_ROI"] = dataset["true_tau_r"] / dataset["true_tau_c"]
        inf_mask = np.isinf(dataset["true_ROI"])
        nan_mask = np.isnan(dataset["true_ROI"])
        dataset["true_ROI"][inf_mask] = np.max(dataset["true_ROI"][~inf_mask])
        dataset["true_ROI"][nan_mask] = 0.0

        # import pdb

        # pdb.set_trace()

        return dataset

    def set_rct_flag(self) -> dict[str, NDArray[Any]]:
        if self.only_rct:
            rct_flag = np.ones(int(self.n_samples * self.rct_ratio))
        else:
            rct_flag = np.zeros(self.n_samples)
            if self.rct_ratio > 0:
                for i in range(0, self.n_samples, int(1 / self.rct_ratio)):
                    rct_flag[i] = 1
        return {"RCT_flag": rct_flag}

    def generate_feature(self) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        if self.only_rct:
            features = np.random.normal(
                size=(int(self.n_samples * self.rct_ratio), self.n_features)
            )
        else:
            features = np.random.normal(size=(self.n_samples, self.n_features))
        return {"features": features}

    def generate_treatment(
        self, features: NDArray[Any], rct_flag: NDArray[Any]
    ) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        T_prob = sigmoid((features[:, 0] - 1.0) / 0.8)
        T_prob = T_prob.clip(0.01, 0.99)
        T_prob[rct_flag == 1] = 0.5
        T: NDArray[Any] = np.random.binomial(1, T_prob).astype(bool)
        treatment_prob = T_prob[T == 1]
        control_prob = T_prob[T == 0]
        plt.clf()
        # plt.hist(T_prob, bins=20, alpha=0.5, label="T_prob")
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
        T_prob: NDArray[Any],
        rct_flag: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        std = self.ps_delta * np.sqrt(np.pi / 2)
        T_prob_pred = T_prob + np.random.normal(0, std, size=len(features))
        T_prob_pred[rct_flag == 1] = 0.5
        T_prob_pred = T_prob_pred.clip(0.01, 0.99)
        return {"T_prob_pred": T_prob_pred}

    # T_ProbとT_Prob_predのRMSEを計算
    def calculate_rmse(
        self,
        T_prob: NDArray[Any],
        T_prob_pred: NDArray[Any],
    ) -> float:
        return float(np.sqrt(np.mean((T_prob - T_prob_pred) ** 2)))

    def conversion_effect(
        self, features: NDArray[Any]
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        baseline_effect = 0.5 * features[:, 0] + features[:, 1] - 3
        interaction_effect_ = 0.5 * features[:, 2]
        interaction_effect = np.exp(interaction_effect_)
        return baseline_effect, interaction_effect

    def visit_effect(self, features: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        baseline_effect = 0.5 * features[:, 0] + features[:, 1] - 4
        interaction_effect_ = (
            0.2 * features[:, 0] + 0.3 * features[:, 1] + 0.6 * features[:, 2]
        )
        interaction_effect = 0.1 * np.exp(interaction_effect_)

        return baseline_effect, interaction_effect

    def generate_conversion(
        self,
        features: NDArray[Any],
        T: Optional[NDArray[Any]] = None,
    ) -> Dict[str, NDArray[Any]]:
        np.random.seed(self.seed)
        baseline_effect, interaction_effect = self.conversion_effect(features)
        if self.train_flg:
            treatment_effect = T * interaction_effect
            prob_purchase = np.clip(
                sigmoid((baseline_effect + treatment_effect)),
                0.01,
                0.99,
            )
            treatment_features = features[T == 1]
            control_features = features[T == 0]
            baseline_effect_treatment, interaction_effect_treatment = (
                self.conversion_effect(treatment_features)
            )
            baseline_effect_control, interaction_effect_control = (
                self.conversion_effect(control_features)
            )
            prob_purchase_treatment = sigmoid(
                (baseline_effect_treatment + interaction_effect_treatment)
            )
            prob_purchase_control = sigmoid((baseline_effect_control))
            prob_purchase_treatment_if_non_treatment = sigmoid(
                (baseline_effect_treatment)
            )
            prob_purchace_control_if_treatment = sigmoid(
                (baseline_effect_control + interaction_effect_control)
            )
            purchase = np.random.binomial(1, prob_purchase)
            plt.clf()
            plt.hist(
                prob_purchase_treatment, bins=20, alpha=0.5, label="Purchase_Treatment"
            )
            plt.hist(
                prob_purchase_control, bins=20, alpha=0.5, label="Purchase_Control"
            )
            plt.hist(
                prob_purchase_treatment_if_non_treatment,
                bins=20,
                alpha=0.5,
                label="if_non_treatment",
            )
            plt.hist(
                prob_purchace_control_if_treatment,
                bins=20,
                alpha=0.5,
                label="if_treatment",
            )
            plt.legend()
            plt.savefig("purchase_prob.png")
        # import pdb

        # pdb.set_trace()
        true_mu_r_1 = np.clip(sigmoid(baseline_effect + interaction_effect), 0.01, 0.99)
        true_mu_r_0 = np.clip(sigmoid(baseline_effect), 0.01, 0.99)
        true_tau_r = true_mu_r_1 - true_mu_r_0
        if self.train_flg:
            return {
                "y_r": purchase,
                "true_mu_r_1": true_mu_r_1,
                "true_mu_r_0": true_mu_r_0,
                "true_tau_r": true_tau_r,
            }
        else:
            return {
                "true_mu_r_1": true_mu_r_1,
                "true_mu_r_0": true_mu_r_0,
                "true_tau_r": true_tau_r,
            }

    def generate_visit(
        self,
        features: NDArray[Any],
        T: Optional[NDArray[Any]] = None,
        y_r: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        np.random.seed(self.seed)

        baseline_effect, interaction_effect = self.visit_effect(features)
        if self.train_flg:
            treatment_effect = T * interaction_effect
            prob_visit = np.clip(
                sigmoid((baseline_effect + treatment_effect)),
                0.01,
                0.99,
            )
            visit = np.where(y_r == 1, 1, np.random.binomial(1, prob_visit))
            treatment_features = features[T == 1]
            control_features = features[T == 0]
            baseline_effect_treatment, interaction_effect_treatment = self.visit_effect(
                treatment_features
            )
            baseline_effect_control, interaction_effect_control = self.visit_effect(
                control_features
            )
            prob_visit_treatment = sigmoid(
                (baseline_effect_treatment + interaction_effect_treatment)
            )
            prob_visit_control = sigmoid((baseline_effect_control))
            prob_visit_treatment_if_non_treatment = sigmoid((baseline_effect_treatment))
            prpb_visit_control_if_treatment = sigmoid(
                (baseline_effect_control + interaction_effect_control),
            )

            plt.clf()
            plt.hist(prob_visit_treatment, bins=20, alpha=0.5, label="Visit_Treatment")
            plt.hist(prob_visit_control, bins=20, alpha=0.5, label="Visit_Control")
            plt.hist(
                prob_visit_treatment_if_non_treatment,
                bins=20,
                alpha=0.5,
                label="if_non_treatment",
            )
            plt.hist(
                prpb_visit_control_if_treatment,
                bins=20,
                alpha=0.5,
                label="if_treatment",
            )
            plt.legend()
            plt.savefig("visit_prob.png")
        # import pdb

        # pdb.set_trace()
        true_mu_c_1 = np.clip(sigmoid(baseline_effect + interaction_effect), 0.01, 0.99)
        true_mu_c_0 = np.clip(sigmoid(baseline_effect), 0.01, 0.99)
        true_tau_c = true_mu_c_1 - true_mu_c_0

        if self.train_flg:
            return {
                "y_c": visit,
                "true_mu_c_1": true_mu_c_1,
                "true_mu_c_0": true_mu_c_0,
                "true_tau_c": true_tau_c,
            }
        else:
            return {
                "true_mu_c_1": true_mu_c_1,
                "true_mu_c_0": true_mu_c_0,
                "true_tau_c": true_tau_c,
            }

    def culculate_doubly_robust(
        self,
        features: NDArray[Any],
        T: NDArray[Any],
        T_prob: NDArray[Any],
        y_r: NDArray[Any],
        y_c: NDArray[Any],
        true_mu_r_1: NDArray[Any],
        true_mu_r_0: NDArray[Any],
        true_mu_c_1: NDArray[Any],
        true_mu_c_0: NDArray[Any],
    ) -> Dict[str, NDArray[Any]]:
        treatment_mask = T == 1
        control_mask = T == 0
        treatment_features = features[treatment_mask]
        control_features = features[control_mask]
        treatment_purchase = y_r[treatment_mask]
        control_purchase = y_r[control_mask]
        treatment_visit = y_c[treatment_mask]
        control_visit = y_c[control_mask]
        # mu_r_0 = LGBMClassifier(verbose=-1, random_state=42).fit(
        #     control_features, control_purchase
        # )
        # mu_r_1 = LGBMClassifier(verbose=-1, random_state=42).fit(
        #     treatment_features, treatment_purchase
        # )
        # mu_c_0 = LGBMClassifier(verbose=-1, random_state=42).fit(
        #     control_features, control_visit
        # )
        # mu_c_1 = LGBMClassifier(verbose=-1, random_state=42).fit(
        #     treatment_features, treatment_visit
        # )
        # mu_r_1_pred = mu_r_1.predict_proba(features)[:, 1]
        # mu_r_0_pred = mu_r_0.predict_proba(features)[:, 1]
        # mu_c_1_pred = mu_c_1.predict_proba(features)[:, 1]
        # mu_c_0_pred = mu_c_0.predict_proba(features)[:, 1]
        # rmse_mu_r_1 = np.sqrt(np.mean((true_mu_r_1 - mu_r_1_pred) ** 2))
        # rmse_mu_r_0 = np.sqrt(np.mean((true_mu_r_0 - mu_r_0_pred) ** 2))
        # rmse_mu_c_1 = np.sqrt(np.mean((true_mu_c_1 - mu_c_1_pred) ** 2))
        # rmse_mu_c_0 = np.sqrt(np.mean((true_mu_c_0 - mu_c_0_pred) ** 2))
        doubly_robust = {}
        # doubly_robust["y_r_dr"] = np.where(
        #     T == 1,
        #     (y_r - true_mu_r_1) / T_prob + true_mu_r_1,
        #     (y_r - true_mu_r_0) / (1 - T_prob) + true_mu_r_0,
        # )
        # doubly_robust["y_c_dr"] = np.where(
        #     T == 1,
        #     (y_c - true_mu_c_1) / T_prob + true_mu_c_1,
        #     (y_c - true_mu_c_0) / (1 - T_prob) + true_mu_c_0,
        # )
        std = self.delta * np.sqrt(np.pi / 2)
        mu_r_1_pred = true_mu_r_1 + np.random.normal(0, std, size=len(features))
        mu_r_0_pred = true_mu_r_0 + np.random.normal(0, std, size=len(features))
        mu_c_1_pred = true_mu_c_1 + np.random.normal(0, std, size=len(features))
        mu_c_0_pred = true_mu_c_0 + np.random.normal(0, std, size=len(features))
        doubly_robust["y_r_dr"] = np.where(
            T == 1,
            (y_r - mu_r_1_pred) / T_prob + mu_r_1_pred,
            (y_r - mu_r_0_pred) / (1 - T_prob) + mu_r_0_pred,
        )
        doubly_robust["y_c_dr"] = np.where(
            T == 1,
            (y_c - mu_c_1_pred) / T_prob + mu_c_1_pred,
            (y_c - mu_c_0_pred) / (1 - T_prob) + mu_c_0_pred,
        )
        # import pdb

        # pdb.set_trace()

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
) -> Tuple[Dict[str, NDArray[Any]], Dict[str, NDArray[Any]]]:
    dataset["strata"] = dataset["T"] * 1.1 + dataset["RCT_flag"]
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset["features"])),
        train_size=0.5,
        random_state=0,
        stratify=dataset["strata"],
    )
    train_dataset = {}
    val_dataset = {}
    for key, value in dataset.items():
        train_dataset[key] = value[train_idx]
        val_dataset[key] = value[val_idx]

    return train_dataset, val_dataset
