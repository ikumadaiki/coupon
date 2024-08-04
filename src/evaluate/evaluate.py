from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import interp1d


def calculate_values(
    roi_scores: NDArray[Any],
    true_tau_r: NDArray[Any],
    true_tau_c: NDArray[Any],
) -> Tuple[Any, Any]:
    sorted_indices = np.argsort(roi_scores)[::-1]
    p_values = np.linspace(0, 1, 50)
    incremental_costs = []
    incremental_values = []

    for p in p_values:
        top_p_indices = sorted_indices[: int(p * len(roi_scores))]

        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(true_tau_r[top_p_indices])
        ATE_Yc = np.mean(true_tau_c[top_p_indices])

        incremental_costs.append(ATE_Yc * len(top_p_indices))
        incremental_values.append(ATE_Yr * len(top_p_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
    # nanがあれば0に変換
    incremental_costs = np.array(incremental_costs)
    incremental_values = np.array(incremental_values)
    incremental_costs[np.isnan(incremental_costs)] = 0
    incremental_values[np.isnan(incremental_values)] = 0
    incremental_costs[0] = 0
    incremental_values[0] = 0

    return incremental_costs, incremental_values


def cost_curve(
    rct_ratio: float,
    incremental_costs: NDArray[Any],
    incremental_values: NDArray[Any],
    label: str,
) -> None:
    normalized_costs = incremental_costs / incremental_costs.max()
    incremental_values += rct_ratio * normalized_costs
    normalized_values = incremental_values / incremental_values.max()

    # グラフ描画

    # 線形補間による関数の定義
    curve_function = interp1d(
        normalized_costs, normalized_values, fill_value="extrapolate"
    )

    # y = x 関数との差を積分（y = x より上の部分のみ）
    def area_above_y_equals_x(x: float) -> float:
        difference = curve_function(x) - x
        return float(difference)

    area, error = quad(area_above_y_equals_x, 0, 1)
    print(
        f"The area above y = x is approximately {area:.4f}, with an error of {error:.4e}."
    )

    plt.plot(normalized_costs, normalized_values, label=f"{label}_{area:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.legend()
    plt.savefig("cost_curve.png")


def cost_curve_alpha(
    rct_ratio: float,
    incremental_costs: NDArray[Any],
    incremental_values: NDArray[Any],
    label: str,
    ps_delta: float,
) -> None:
    normalized_costs = incremental_costs / incremental_costs.max()
    incremental_values += rct_ratio * normalized_costs
    normalized_values = incremental_values / incremental_values.max()

    # 線形補間による関数の定義
    curve_function = interp1d(
        normalized_costs, normalized_values, fill_value="extrapolate"
    )

    # y = x 関数との差を積分（y = x より上の部分のみ）
    def area_above_y_equals_x(x: float) -> float:
        difference = curve_function(x) - x
        return float(difference)

    area, error = quad(area_above_y_equals_x, 0, 1)
    print(
        f"The area above y = x is approximately {area:.4f}, with an error of {error:.4e}."
    )

    plt.plot(normalized_costs, normalized_values, label=f"alpha={label}_({area:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.legend()
    plt.savefig(f"cost_curve_ps_delta={ps_delta}.png")


def optimize_alpha(
    rct_ratio: float,
    roi_scores: NDArray[Any],
    true_tau_r: NDArray[Any],
    true_tau_c: NDArray[Any],
) -> Tuple[Any, Any]:
    random_treatment_indices = int(len(roi_scores) * rct_ratio)
    model_based_treatment = roi_scores[: (len(roi_scores) - random_treatment_indices)]
    sorted_indices = np.argsort(model_based_treatment)[::-1]
    p_values = np.linspace(0, 1, 50)
    incremental_costs = []
    incremental_values = []

    for p in p_values:
        top_p_indices = sorted_indices[: int(p * len(model_based_treatment))]

        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(true_tau_r[top_p_indices])
        ATE_Yc = np.mean(true_tau_c[top_p_indices])

        incremental_costs.append(ATE_Yc * len(top_p_indices))
        incremental_values.append((1 - rct_ratio) * ATE_Yr * len(top_p_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
    # nanがあれば0に変換
    incremental_costs = np.array(incremental_costs)
    incremental_values = np.array(incremental_values)
    incremental_costs[np.isnan(incremental_costs)] = 0
    incremental_values[np.isnan(incremental_values)] = 0
    incremental_costs[0] = 0
    incremental_values[0] = 0

    return incremental_costs, incremental_values


# # 例として適用するデータ
# incremental_costs = np.array([0, 0.25, 0.5, 0.75, 1])
# incremental_values = np.array([0, 0.3, 0.6, 0.8, 1])
# cost_curve(incremental_costs, incremental_values, "Example Curve")
