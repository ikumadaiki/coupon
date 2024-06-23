import init_data

# 必要なライブラリのインポート
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import SLearner
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor

# データの分割
def split_data(df, seed):
    X = df.drop(['treatment', 'exposure', 'visit', 'conversion'], axis=1)
    X = (X - X.mean()) / X.std()
    T = df['treatment']
    y_r = df['conversion']
    y_c = df['visit']

    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test = train_test_split(
        X, T, y_r, y_c, train_size=0.7, random_state=seed, stratify=T
    )

    # インデックスをリセット
    T_test = T_test.reset_index(drop=True)
    y_r_test = y_r_test.reset_index(drop=True)
    y_c_test = y_c_test.reset_index(drop=True)

    return X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test

def custom_objective(y_pred: np.ndarray, train_data: lgb.Dataset, y_r_train, y_c_train, T_train):
    y_r = y_r_train
    y_c = y_c_train
    t = T_train.values
    treatment = (t == 1)
    control = (t == 0)

    N_1 = np.sum(treatment)
    N_0 = np.sum(control)

    p = 1 / (1 + np.exp(-y_pred))

    grad = ((-1) ** t) * (y_r - y_c * p) / np.where(treatment, N_1, N_0)

    hess = y_c * p * (1 - p) / np.where(treatment, N_1, N_0)

    return grad, hess

def get_roi_direct():
    dtrain = lgb.Dataset(X_train, free_raw_data=False)

    params = {
        'objective': lambda y_pred, train_data: custom_objective(y_pred, train_data),
        'verbose': -1,
    }
    bst = lgb.train(params, dtrain)
    roi_direct = bst.predict(X_test, num_iteration=bst.best_iteration)
    return roi_direct
    
def get_roi_tpmsl():


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

def calculate_values(roi_scores):
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

def main():
    df = init_data.get_data()
    # パラメータの設定
    seed = 42
    X_train, X_test, T_train, T_test, y_r_train, y_r_test, y_c_train, y_c_test = split_data(df, seed)