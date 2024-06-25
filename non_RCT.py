import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import SLearner
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n, p, seed=42):
    np.random.seed(seed)
    features = np.random.normal(size=(n, p))
    df = pd.DataFrame(features, columns=[f"x_{i}" for i in range(p)])
    x_cols = df.columns.to_list()
    return df, x_cols

def generate_treatment(df, x_cols, seed=42):
    np.random.seed(seed)
    logistic_model = LogisticRegression(max_iter=1000)
    df["target"] = (np.dot(df[x_cols].values, np.random.uniform(0.1, 0.5, size=len(x_cols))) - 0.5 + np.random.normal(0, 0.5, size = len(df)) > 0).astype(int)
    logistic_model.fit(df[x_cols], df["target"])
    df["T_prob"] = logistic_model.predict_proba(df[x_cols])[:, 1]
    df["T_prob"] = df["T_prob"].clip(0.01, 0.99)
    df["T"] = np.random.binomial(1, df["T_prob"])
    df.drop("target", axis=1, inplace=True)
    return df

def generate_visit(df, x_cols, seed=42):
    np.random.seed(seed)
    interaction_effects = sigmoid(np.sum(df.iloc[:, :len(x_cols)], axis=1))
    baseline_effect = 0.3 + df['x_2'] * 0.4 + df["x_4"] * 0.1
    treatment_effect = df['T'] * (0.4 + interaction_effects)
    prob_visit = np.clip(baseline_effect + treatment_effect, 0, 1)
    df['visit'] = np.random.binomial(1, prob_visit)
    return df

def generate_conversion(df, x_cols, seed=42):
    np.random.seed(seed)
    interaction_effects_purchase = sigmoid(np.sum(df.iloc[:, :len(x_cols)], axis=1))
    baseline_effect_purchase = 0.1 + df['x_5'] * 0.3 + df["x_7"] * 0.3
    treatment_effect_purchase = df['T'] * (0.2 + interaction_effects_purchase)
    prob_purchase = np.clip(baseline_effect_purchase + treatment_effect_purchase, 0, 1)
    df['purchase'] = np.where(df['visit'] == 1, np.random.binomial(1, prob_purchase), 0)
    return df

def predict_outcome(df, x_cols):
    df_t0 = df[df["T"]==0]
    df_t1 = df[df["T"]==1]
    mu_r_0 = LGBMRegressor().fit(df_t0[x_cols], df_t0["purchase"])
    mu_r_1 = LGBMRegressor().fit(df_t1[x_cols], df_t1["purchase"])
    mu_c_0 = LGBMRegressor().fit(df_t0[x_cols], df_t0["visit"])
    mu_c_1 = LGBMRegressor().fit(df_t1[x_cols], df_t1["visit"])
    return mu_r_0, mu_r_1, mu_c_0, mu_c_1

def preprocess_data(df, x_cols, mu_r_0, mu_r_1, mu_c_0, mu_c_1):
    X, T, y_r, y_c = df[x_cols], df["T"], df["purchase"], df["visit"]
    df["y_r_ipw"] = np.where(df["T"]==1, df["purchase"] / df["T_prob"], df["purchase"] / (1 - df["T_prob"]))
    df["y_c_ipw"] = np.where(df["T"]==1, df["visit"] / df["T_prob"], df["visit"] / (1 - df["T_prob"]))
    df["y_c_dr"] = np.where(T==1, (df["visit"] - mu_c_1.predict(X)) / df["T_prob"] + mu_c_1.predict(X), (df["visit"] - mu_c_0.predict(X)) / (1 - df["T_prob"]) + mu_c_0.predict(X))
    df["y_r_dr"] = np.where(T==1, (df["purchase"] - mu_r_1.predict(X)) / df["T_prob"] + mu_r_1.predict(X), (df["purchase"] - mu_r_0.predict(X)) / (1 - df["T_prob"]) + mu_r_0.predict(X))
    return df, X, T, y_r, y_c

def split_data(df, x_cols, X, T, y_r, y_c):
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_dr_train, y_r_dr_test, y_c_dr_train, y_c_dr_test, y_r_ipw_train, y_r_ipw_test, y_c_ipw_train, y_c_ipw_test = train_test_split(
        X, T, y_r, y_c, df["y_r_dr"], df["y_c_dr"], df["y_r_ipw"], df["y_c_ipw"], train_size=0.7, random_state=0, stratify=T
        )

    T_test = T_test.reset_index(drop=True)
    y_r_test = y_r_test.reset_index(drop=True)
    y_c_test = y_c_test.reset_index(drop=True)

    return X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_dr_train, y_r_dr_test, y_c_dr_train, y_c_dr_test, y_r_ipw_train, y_r_ipw_test, y_c_ipw_train, y_c_ipw_test

def create_custom_objective(y_r_, y_c_, T_train):
    def custom_objective(y_pred: np.ndarray, train_data: lgb.Dataset):
        y_r = y_r_
        y_c = y_c_
        t = T_train.values
        treatment = (t == 1)
        control = (t == 0)

        N_1 = np.sum(treatment)
        N_0 = np.sum(control)

        p = 1 / (1 + np.exp(-y_pred))

        grad = ((-1) ** t) * (y_r - y_c * p) / np.where(treatment, N_1, N_0)

        hess = y_c * p * (1 - p) / np.where(treatment, N_1, N_0)

        # grad = -(tau_r - tau_c * p) / (N_1 + N_0)
        # hess = tau_c * p * (1 - p) / (N_1 + N_0)

        return grad, hess
    return custom_objective

def get_roi_direct(X_train, custom_objective, X_test):
    dtrain = lgb.Dataset(X_train, free_raw_data=False)

    params = {
        'objective': lambda y_pred, train_data: custom_objective(y_pred, train_data),
        'verbose': -1,
    }
    bst = lgb.train(params, dtrain)
    roi_direct = bst.predict(X_test, num_iteration=bst.best_iteration)
    roi_direct = 1 / (1 + np.exp(-roi_direct))
    return roi_direct

def get_roi_tpmsl(y_r_train, y_c_train, X_train, X_test, T_train):
    # モデルの構築
    models = LGBMRegressor()
    # models = LinearRegression()
    S_learner_r = SLearner(overall_model = models)
    S_learner_r.fit(y_r_train, T_train, X = X_train)

    S_learner_c = SLearner(overall_model = models)
    S_learner_c.fit(y_c_train, T_train, X = X_train)

    # 効果の推定
    tau_r = S_learner_r.effect(X_test)
    tau_c = S_learner_c.effect(X_test)
    roi_tpmsl = tau_r / tau_c

    scaler = MinMaxScaler()
    roi_tpmsl = scaler.fit_transform(roi_tpmsl.reshape(-1, 1)).flatten()

    # roi_tpmsl = sigmoid(roi_tpmsl)
    return roi_tpmsl

def calculate_values(roi_scores, T_test, y_r_test, y_c_test):
    sorted_indices = np.argsort(roi_scores)[::-1]
    p_values = np.linspace(0, 1, 30)
    incremental_costs = []
    incremental_values = []
    
    for p in p_values:
        top_p_indices = sorted_indices[:int(p * len(roi_scores))]
        treatment_indices = (T_test[top_p_indices] == 1)
        
        # ATE (Average Treatment Effect) の計算
        ATE_Yr = np.mean(y_r_test[top_p_indices][treatment_indices]) - np.mean(y_r_test[top_p_indices][~treatment_indices])
        ATE_Yc = np.mean(y_c_test[top_p_indices][treatment_indices]) - np.mean(y_c_test[top_p_indices][~treatment_indices])
        
        incremental_costs.append(ATE_Yc * np.sum(treatment_indices))
        incremental_values.append(ATE_Yr * np.sum(treatment_indices))
        # print(ATE_Yr , ATE_Yc,np.sum(treatment_indices))
        incremental_costs[0] = 0
        incremental_values[0] = 0
        
    return incremental_costs, incremental_values

def plot_cost_curve(incremental_costs_direct_dr, incremental_values_direct_dr, incremental_costs_direct_ipw, incremental_values_direct_ipw, incremental_costs_direct_naive, incremental_values_direct_naive, incremental_costs_tpmsl, incremental_values_tpmsl):
    plt.figure(figsize=(10, 6))
    plt.plot(incremental_costs_direct_dr, incremental_values_direct_dr, label="DR")
    plt.plot(incremental_costs_direct_ipw, incremental_values_direct_ipw, label="IPW")
    plt.plot(incremental_costs_direct_naive, incremental_values_direct_naive, label="Naive")
    plt.plot(incremental_costs_tpmsl, incremental_values_tpmsl, label="TPMSL")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Cost Curve Comparison')
    plt.xlabel('Incremental Cost')
    plt.ylabel('Incremental Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():    
    seed = 42
    n = 10_000_000
    p = 10
    df, x_cols = generate_data(n, p, seed)
    df = generate_treatment(df, x_cols, seed)
    df = generate_visit(df, x_cols)
    df = generate_conversion(df, x_cols)
    mu_r_0, mu_r_1, mu_c_0, mu_c_1 = predict_outcome(df, x_cols)
    df, X, T, y_r, y_c = preprocess_data(df, x_cols, mu_r_0, mu_r_1, mu_c_0, mu_c_1)
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test, y_r_dr_train, y_r_dr_test, y_c_dr_train, y_c_dr_test, y_r_ipw_train, y_r_ipw_test, y_c_ipw_train, y_c_ipw_test = split_data(df, x_cols, X, T, y_r, y_c)
    custom_objective_dr = create_custom_objective(y_r_dr_train, y_c_dr_train, T_train)
    custom_objective_ipw = create_custom_objective(y_r_ipw_train, y_c_ipw_train, T_train)
    custom_objective_naive = create_custom_objective(y_r_train, y_c_train, T_train)
    roi_direct_dr = get_roi_direct(X_train, custom_objective_dr, X_test)
    roi_direct_ipw = get_roi_direct(X_train, custom_objective_ipw, X_test)
    roi_direct_naive = get_roi_direct(X_train, custom_objective_naive, X_test)
    roi_tpmsl = get_roi_tpmsl(y_r_train, y_c_train, X_train, X_test, T_train)
    incremental_costs_direct_dr, incremental_values_direct_dr = calculate_values(roi_direct_dr, T_test, y_r_test, y_c_test)
    incremental_costs_direct_ipw, incremental_values_direct_ipw = calculate_values(roi_direct_ipw, T_test, y_r_test, y_c_test)
    incremental_costs_direct_naive, incremental_values_direct_naive = calculate_values(roi_direct_naive, T_test, y_r_test, y_c_test)
    incremental_costs_tpmsl, incremental_values_tpmsl = calculate_values(roi_tpmsl, T_test, y_r_test, y_c_test)
    plot_cost_curve(incremental_costs_direct_dr, incremental_values_direct_dr, incremental_costs_direct_ipw, incremental_values_direct_ipw, incremental_costs_direct_naive, incremental_values_direct_naive, incremental_costs_tpmsl, incremental_values_tpmsl)