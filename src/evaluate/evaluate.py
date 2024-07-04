from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def calculate_values(
    roi_scores: NDArray[np.float_],
    T_test: NDArray[np.float_],
    y_r_test: NDArray[np.float_],
    y_c_test: NDArray[np.float_],
) -> Tuple:
    sorted_indices = np.argsort(roi_scores)[::-1]
    p_values = np.linspace(0, 1, 50)
    incremental_costs = []
    incremental_values = []

    for p in p_values:
        top_p_indices = sorted_indices[: int(p * len(roi_scores))]
        treatment_indices = T_test[top_p_indices] == 1

        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(y_r_test[top_p_indices][treatment_indices]) - np.mean(
            y_r_test[top_p_indices][~treatment_indices]
        )
        ATE_Yc = np.mean(y_c_test[top_p_indices][treatment_indices]) - np.mean(
            y_c_test[top_p_indices][~treatment_indices]
        )

        incremental_costs.append(ATE_Yc * np.sum(treatment_indices))
        incremental_values.append(ATE_Yr * np.sum(treatment_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
        incremental_costs[0] = 0
        incremental_values[0] = 0

    return incremental_costs, incremental_values


def cost_curve(
    incremental_costs: NDArray[np.float_], incremental_values: NDArray[np.float_]
) -> None:
    plt.plot(
        incremental_costs / max(incremental_costs),
        incremental_values / max(incremental_values),
    )
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.savefig("cost_curve.png")
