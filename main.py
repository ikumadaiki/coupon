import matplotlib.pyplot as plt

from src.evaluate.evaluate import calculate_values
from src.prediction.model_direct import get_loss, loader
from src.prediction.predict import get_roi, get_roi_tpmsl
from src.preprocess.make_data import (
    generate_conversion,
    generate_feature,
    generate_treatment,
    generate_visit,
    predict_outcome,
    predict_treatment,
    preprocess_data,
    split_data,
)


def main(predict_ps: bool) -> None:
    seed = 42
    n = 100_000
    p = 8
    dic = {}
    std = 1.0
    num_epochs = 200
    lr = 0.0001
    dic = generate_feature(n, p, dic, seed)
    dic = generate_treatment(dic, seed)
    if predict_ps:
        dic = predict_treatment(dic)
    dic = generate_visit(dic, std, seed)
    dic = generate_conversion(dic, std, seed)
    mu_r_0, mu_r_1, mu_c_0, mu_c_1 = predict_outcome(dic)
    dic = preprocess_data(dic, mu_r_0, mu_r_1, mu_c_0, mu_c_1)
    (
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
    ) = split_data(dic, seed)
    method_dic = {
        # "Direct": [y_r_train, y_c_train, y_r_val, y_c_val],
        # "IPW": [y_r_ipw_train, y_c_ipw_train, y_r_ipw_val, y_c_ipw_val],
        "DR": [y_r_dr_train, y_c_dr_train, y_r_dr_val, y_c_dr_val],
    }
    roi_dic = {}
    for method in method_dic:
        dl, dl_val = loader(
            X_train,
            T_train,
            method_dic[method][0],
            method_dic[method][1],
            X_val,
            T_val,
            method_dic[method][2],
            method_dic[method][3],
            seed,
        )
        model, loss_history, loss_history_val = get_loss(
            num_epochs, lr, X_train, dl, dl_val
        )
        # plot_loss(loss_history, loss_history_val)
        roi = get_roi(model, X_test)
        roi_dic[method] = roi

    roi_tpmsl = get_roi_tpmsl(X_train, y_r_train, y_c_train, T_train, X_test)
    roi_dic["TPMSL"] = roi_tpmsl
    import pdb

    pdb.set_trace()
    plt.clf()
    for roi in roi_dic:
        incremental_costs, incremental_values = calculate_values(
            roi_dic[roi], T_test, y_r_test, y_c_test
        )
        plt.plot(
            incremental_costs / max(incremental_costs),
            incremental_values / max(incremental_values),
            label=roi,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Incremental Costs")
    plt.ylabel("Incremental Values")
    plt.legend()
    plt.savefig("cost_curve.png")


if __name__ == "__main__":
    main(predict_ps=True)
